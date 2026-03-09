import json
import re


def extract_answer(text):
    """
    提取最后一个 #### 的数字
    如果没有找到返回 None
    """
    if text is None:
        return None

    matches = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text)

    if len(matches) == 0:
        return None

    return matches[-1]


# 读取结果文件
with open("gsm_symbolic_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

original_results = data["original_results"]

# 统计变量
original_total = 0
original_correct = 0

new_total = 0
new_correct = 0


for item in original_results:

    # ---------- 原始问题 ----------
    original_model_answer = item.get("original_model_answer")

    # 提取模型最终答案
    model_ans = extract_answer(original_model_answer)

    # 如果没有 ####，说明被截断或没输出最终答案
    if model_ans is None:
        continue

    original_total += 1

    if item.get("original_is_correct"):
        original_correct += 1


    # ---------- 新问题 ----------
    new_questions = item.get("new_questions_results", [])

    for q in new_questions:

        model_answer = q.get("model_answer")

        new_ans = extract_answer(model_answer)

        # 如果没有 #### 就跳过
        if new_ans is None:
            continue

        new_total += 1

        if q.get("is_correct"):
            new_correct += 1


# 计算准确率
original_accuracy = original_correct / original_total if original_total > 0 else 0
new_accuracy = new_correct / new_total if new_total > 0 else 0


# 输出结果
print("=" * 60)
print("原始问题统计（只统计成功输出 #### 的样本）")
print(f"总数: {original_total}")
print(f"正确数: {original_correct}")
print(f"准确率: {original_accuracy:.2%}")

print("\n新问题统计（只统计成功输出 #### 的样本）")
print(f"总数: {new_total}")
print(f"正确数: {new_correct}")
print(f"准确率: {new_accuracy:.2%}")
print("=" * 60)