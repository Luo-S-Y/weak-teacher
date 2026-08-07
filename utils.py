"""通用工具: 答案提取 / 数学判题 / 文本组装 (复用 0726-deepseek-r1 的判题逻辑)"""
import json
import re


def extract_answer(text):
    """从模型输出中提取最终答案 (优先 \\boxed, 其次最后数字/表达式)。"""
    boxed = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    if boxed:
        return boxed[-1].strip()
    lines = text.strip().split('\n')
    for line in reversed(lines):
        m = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', line)
        if m:
            return m.group(1).strip()
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums:
        return nums[-1]
    return ""


def _frac_to_plain(s):
    """\frac{a}{b} -> (a)/(b), 循环展开至无嵌套"""
    while True:
        m = re.search(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', s)
        if not m:
            return s
        s = s[:m.start()] + f"({m.group(1)})/({m.group(2)})" + s[m.end():]


def _as_number(s):
    """尝试把 '1/2' / '\\frac{1}{2}' 转成 float (Fraction), 失败返回 None"""
    from fractions import Fraction
    try:
        return float(Fraction(_frac_to_plain(s.strip()).replace("(", "").replace(")", "")))
    except Exception:
        return None


def answers_match(pred, expected, tolerance=1e-2):
    """数值/符号表达式判等 (分数/小数/LaTeX 分数/符号表达式)。"""
    p, e = pred.strip(), expected.strip()
    if not p or not e:
        return False
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < tolerance
    except ValueError:
        pass
    a, b = _as_number(p), _as_number(e)
    if a is not None and b is not None:
        return abs(a - b) < tolerance
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        diff = simplify(parse_latex(p) - parse_latex(e))
        if diff == 0:
            return True
        try:
            return abs(float(diff.evalf())) < tolerance
        except Exception:
            return False
    except Exception:
        return False


def gsm8k_gold_answer(answer):
    """GSM8K 标准答案 '... The answer is 42' -> 42"""
    m = re.search(r'####\s*(.+)$', answer)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'-?\d+\.?\d*', answer)
    return nums[-1] if nums else ""


def is_correct_by_rule(teacher_answer, gold_answer):
    """E3 规则验证器: 教师答案是否与标准答案一致"""
    return answers_match(teacher_answer, gold_answer)


def build_messages(problem, output=None, system="You are a helpful math assistant. Reason step by step and put the final answer in \\boxed{}."):
    """Qwen chat 消息格式 (Qwen2.5 教师生成 / Qwen3 学生训练通用)。"""
    if output is None:
        return [{"role": "system", "content": system}, {"role": "user", "content": problem}]
    return [{"role": "system", "content": system},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": output}]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def log(msg):
    import time
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
