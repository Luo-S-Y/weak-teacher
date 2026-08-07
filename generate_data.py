"""Step 1: 教师数据生成 (GSM8K 训练集)
- 主教师 Qwen2.5-0.5B: 生成 CoT 轨迹 + 每 token logprob (E1)
- 自一致性: 主教师 K 次采样答案 (E2)
- 规则判题: 与 GSM8K 标准答案比对 (E3)
- 学生基线: Qwen3-0.7B 生成答案 (E4, on-policy 近似)
- 辅助教师 Qwen2.5-1.5B: 投票答案 (E5)
输出: data/generated/gsm8k.jsonl (已存在则跳过)

用法: python generate_data.py  (SKIP_SC=1 可跳过自一致性加速)
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from utils import (log, save_jsonl, extract_answer,
                   is_correct_by_rule, build_messages)

OUT = os.path.join(C.GEN_DIR, "gsm8k.jsonl")


def get_tokenizer(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    return model


def make_prompt(tok, problem, model_id, student=False):
    msgs = build_messages(problem)
    kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if student else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)


def batch_generate(model, tok, prompts, max_new, temp, top_p, record_logprob=False):
    """批量生成; 返回 [(text, logprobs_or_None), ...]"""
    import torch
    batch = 32
    results = []
    for i in range(0, len(prompts), batch):
        ps = prompts[i:i + batch]
        inputs = tok(ps, return_tensors="pt", padding=True, truncation=True,
                     max_length=C.MAX_PROBLEM_LEN).to("cuda")
        do_sample = temp > 0
        kwargs = dict(max_new_tokens=max_new, do_sample=do_sample,
                      temperature=max(temp, 1e-6), top_p=top_p,
                      pad_token_id=tok.pad_token_id)
        if record_logprob:
            out = model.generate(**inputs, return_dict_in_generate=True,
                                 output_scores=True, **kwargs)
            seqs, scores = out.sequences, out.scores  # scores[t]: (B, V) logits
            for b in range(len(ps)):
                gen_ids = seqs[b][inputs.input_ids.shape[1]:]
                lps = []
                for t, tid in enumerate(gen_ids):
                    lp = torch.log_softmax(scores[t][b], dim=-1)[tid].item()
                    lps.append(lp)
                results.append((tok.decode(gen_ids, skip_special_tokens=True), lps))
        else:
            out = model.generate(**inputs, **kwargs)
            for b in range(len(ps)):
                gen_ids = out[b][inputs.input_ids.shape[1]:]
                results.append((tok.decode(gen_ids, skip_special_tokens=True), None))
    return results


def gen_answers(model, tok, problems, model_id, temp, max_new=C.GEN_MAX_NEW):
    """对 problems 逐条生成答案, 返回 answer 列表 (仅提取最终答案, 不存轨迹)"""
    prompts = [make_prompt(tok, p, model_id, student="Qwen3" in model_id) for p in problems]
    out = batch_generate(model, tok, prompts, max_new, temp, C.GEN_TOP_P)
    return [extract_answer(t) for t, _ in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sc", action="store_true", help="跳过 E2 自一致性生成")
    args = ap.parse_args()
    skip_sc = args.skip_sc or os.environ.get("SKIP_SC") == "1"

    if os.path.exists(OUT):
        log(f"已完成: {OUT} (删除可重新生成)"); return

    # ---------- 1. GSM8K (读 prepare_data.py 缓存) ----------
    raw_path = os.path.join(C.RAW_DIR, "gsm8k_train.json")
    if not os.path.exists(raw_path):
        log("未找到 gsm8k_train.json, 先运行: python prepare_data.py")
        sys.exit(1)
    log(f"加载 GSM8K 训练集: {raw_path}")
    tok_main = get_tokenizer(C.TEACHER_MAIN)
    problems, golds = [], []
    for row in json.load(open(raw_path)):
        if len(tok_main(row["problem"]).input_ids) <= C.MAX_PROBLEM_LEN:
            problems.append(row["problem"])
            golds.append(row["gold"])
    log(f"有效问题: {len(problems)}")

    # ---------- 2. 主教师生成 (含 logprob) ----------
    log(f"主教师 {C.TEACHER_MAIN} 生成 CoT + logprob")
    model_main = get_model(C.TEACHER_MAIN)
    prompts = [make_prompt(tok_main, p, C.TEACHER_MAIN) for p in problems]
    main_out = batch_generate(model_main, tok_main, prompts, C.GEN_MAX_NEW,
                              C.GEN_TEMP, C.GEN_TOP_P, record_logprob=True)

    rows = []
    for (p, g, (text, lps)) in zip(problems, golds, main_out):
        ans = extract_answer(text)
        rows.append({"problem": p, "gold": g, "teacher_output": text, "answer": ans,
                     "correct": is_correct_by_rule(ans, g),
                     "avg_logprob": sum(lps) / len(lps) if lps else 0.0,
                     "token_logprobs": lps})
    del model_main
    import torch, gc; gc.collect()
    log(f"主教师判题正确率: {sum(r['correct'] for r in rows)}/{len(rows)} = "
        f"{sum(r['correct'] for r in rows)/len(rows)*100:.1f}%")

    # ---------- 3. 自一致性 E2 (主教师 K 次采样) ----------
    if not skip_sc and C.SELF_CONSISTENCY_K > 1:
        log(f"自一致性: 主教师采样 K={C.SELF_CONSISTENCY_K}")
        model_sc = get_model(C.TEACHER_MAIN)
        K = C.SELF_CONSISTENCY_K
        sc_answers = [[] for _ in rows]
        for k in range(K):
            ans = gen_answers(model_sc, tok_main, problems, C.TEACHER_MAIN,
                              C.SELF_CONSISTENCY_TEMP)
            for i, a in enumerate(ans):
                sc_answers[i].append(a)
            log(f"  round {k+1}/{K} done")
        del model_sc; gc.collect()
        for r, ans in zip(rows, sc_answers):
            from collections import Counter
            cnt = Counter(a for a in ans if a)
            mode, c = (cnt.most_common(1)[0] if cnt else ("", 0))
            r["sc_mode"] = mode
            r["sc_agree"] = (c / K) if cnt else 0.0
            r["sc_answers"] = ans
        log(f"自一致性平均一致率: {sum(r['sc_agree'] for r in rows)/len(rows):.3f}")
    else:
        for r in rows:
            r["sc_mode"], r["sc_agree"], r["sc_answers"] = r["answer"], 1.0, [r["answer"]]

    # ---------- 4. 辅助教师 E5 ----------
    log(f"辅助教师 {C.TEACHER_EXTRA} 生成投票答案")
    tok_extra = get_tokenizer(C.TEACHER_EXTRA)
    model_extra = get_model(C.TEACHER_EXTRA)
    extra_ans = gen_answers(model_extra, tok_extra, problems, C.TEACHER_EXTRA,
                            C.EXTRA_TEACHER_TEMP)
    for r, a in zip(rows, extra_ans):
        r["extra_answer"] = a
        r["extra_match"] = r["answer"] == a and a != ""
    del model_extra; gc.collect()

    # ---------- 5. 学生基线 E4 ----------
    log(f"学生基线 {C.STUDENT_MODEL} 生成答案 (on-policy 近似)")
    tok_stu = get_tokenizer(C.STUDENT_MODEL)
    model_stu = get_model(C.STUDENT_MODEL)
    stu_ans = gen_answers(model_stu, tok_stu, problems, C.STUDENT_MODEL,
                          C.STUDENT_ANS_TEMP)
    for r, a in zip(rows, stu_ans):
        r["student_answer"] = a
        r["student_match"] = r["answer"] == a and a != ""
    del model_stu; gc.collect()

    save_jsonl(OUT, rows)
    log(f"完成 -> {OUT} ({len(rows)} 条)")
    from collections import Counter
    print("correct:", sum(r['correct'] for r in rows), "| sc_agree>0.5:",
          sum(1 for r in rows if r['sc_agree'] > 0.5), "| extra_match:",
          sum(r['extra_match'] for r in rows), "| student_match:",
          sum(r['student_match'] for r in rows), flush=True)


if __name__ == "__main__":
    main()
