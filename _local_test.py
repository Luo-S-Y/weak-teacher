"""本地逻辑验证 (不依赖 GPU): 判题器 + 置信度估计器"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import extract_answer, answers_match, gsm8k_gold_answer, is_correct_by_rule

# 1. boxed 提取
assert extract_answer("思考...\\boxed{42}") == "42"
assert extract_answer("The answer is 32.") == "32"
assert extract_answer("no answer") == ""
# 2. 判等: 分数/数值
assert answers_match("1/2", "\\frac{1}{2}")
assert answers_match("3.0", "3")
assert not answers_match("5", "4")
assert not answers_match("", "4")
# 3. GSM8K gold
assert gsm8k_gold_answer("John has 3 apples. #### 3") == "3"
assert is_correct_by_rule("42", "42")
print("== math utils OK ==")

# 4. 置信度分布 (合成数据)
import estimate, math, numpy as np
rows = [
    {"avg_logprob": -0.1, "sc_agree": 1.0, "correct": True,  "student_match": True,  "extra_match": True},
    {"avg_logprob": -2.0, "sc_agree": 0.25, "correct": False, "student_match": False, "extra_match": False},
    {"avg_logprob": -0.7, "sc_agree": 0.5, "correct": True,  "student_match": False, "extra_match": True},
]
cs = estimate.compute_confidence(rows)
for e in estimate.C.ESTIMATORS:
    assert all(0.0 <= x <= 1.0 for x in cs[e]), (e, cs[e])
print("E0:", cs["E0"])
print("E1:", [round(x, 3) for x in cs["E1"]])
print("E3:", cs["E3"], "| E5:", cs["E5"], "| E6:", [round(x, 3) for x in cs["E6"]])
assert cs["E1"][0] > cs["E1"][1]
assert cs["E5"][2] == 1.0 and cs["E5"][1] == 0.5
assert abs(cs["E6"][0] - 0.7 * 1 - 0.3 * cs["E1"][0]) < 1e-6
print("== confidence estimators OK ==")

# 5. W2 阈值 / W5 课程排序
import config as C
c = np.array(cs["E3"], dtype=np.float32)
idx2 = np.where(c > C.W2_TAU)[0]
idx5 = np.argsort(-c)
assert idx2.tolist() == [0, 2], idx2
assert idx5[0] in (0, 2)
# 6. W5: c 降序权重
sw5 = c[idx5]
assert all(sw5[i] >= sw5[i + 1] for i in range(len(sw5) - 1))
print("== W2/W5 index logic OK ==")

# 7. 42 组合命名唯一
runs = C.all_runs()
assert len(runs) == 42 and len(set(runs)) == 42
print("== 42 runs OK ==")
print("ALL LOCAL TESTS PASSED")
