import json
import re
from datetime import datetime
from generation import Llama


# =========================================================
# 工具函数
# =========================================================

def extract_cot(text: str) -> str:
    """提取 #### 之前的推理过程"""
    if not text:
        return ""
    return text.split("####")[0].strip()


def extract_number(text: str):
    """统一提取 #### 后面的数字"""
    if not text:
        return None
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    return None


def build_prompt(question: str, cot: str) -> str:
    return f"""You are given a question and its correct reasoning process.

Follow the reasoning strictly and compute the final answer.

Question:
{question}

Correct reasoning process:
{cot}

IMPORTANT:
- Do NOT change the reasoning
- Do NOT add new reasoning
- Only compute the final answer
- Output format must be:

#### <number>
"""


# =========================================================
# 加载模型
# =========================================================

generator = Llama.build(
    ckpt_dir="checkpoints/Meta-Llama-3.1-8B-Instruct/original",
    tokenizer_path="checkpoints/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model",
    max_seq_len=4096,
    max_batch_size=1,
)

print("模型加载完成!")


# =========================================================
# 读取新 JSON 结构
# =========================================================

with open("test_results_20260226_144020.json", "r", encoding="utf-8") as f:
    data = json.load(f)

incorrect_samples = []

for item in data["results"]: 

    original_id = item["original_id"]

    for new_q in item["new_questions_results"]:

        if not new_q["is_correct"]:   # 只拿错误样本

            incorrect_samples.append({
                "original_id": original_id,
                "question": new_q["question"],
                "expected_full_answer": new_q["expected_answer"],
                "original_model_answer": new_q["model_answer"]
            })

print(f"\n找到 {len(incorrect_samples)} 个错误样本")
print("=" * 80)


# =========================================================
# 修正流程
# =========================================================

corrected_results = {
    "total_incorrect_samples": len(incorrect_samples),
    "corrected_samples": []
}

for idx, sample in enumerate(incorrect_samples):

    print(f"\n处理样本 {idx+1}/{len(incorrect_samples)}")

    question = sample["question"]
    expected_full = sample["expected_full_answer"]
    original_model_answer = sample["original_model_answer"]

    expected_cot = extract_cot(original_model_answer)
    expected_num = extract_number(expected_full)
    original_num = extract_number(original_model_answer)

    prompt = build_prompt(question, expected_cot)

    try:
        outputs = generator.text_completion(
            [prompt],
            max_gen_len=512,
            temperature=0.0,
            top_p=1.0,
        )
        corrected_answer = outputs[0]["generation"] if outputs else ""
    except Exception as e:
        print(f"推理失败: {e}")
        corrected_answer = None

    corrected_num = extract_number(corrected_answer)

    is_corrected = (
        expected_num is not None and
        corrected_num is not None and
        float(expected_num) == float(corrected_num)
    )

    print(f"期望: {expected_num}")
    print(f"原答案: {original_num}")
    print(f"修正后: {corrected_num}")
    print(f"修正成功: {'✓' if is_corrected else '✗'}")

    corrected_results["corrected_samples"].append({
        "original_id": sample["original_id"],
        "question": question,
        "expected_number": expected_num,
        "original_number": original_num,
        "corrected_number": corrected_num,
        "is_corrected": is_corrected
    })


# =========================================================
# 统计
# =========================================================

corrected_count = sum(
    1 for s in corrected_results["corrected_samples"]
    if s["is_corrected"]
)

success_rate = (
    corrected_count / len(incorrect_samples)
    if incorrect_samples else 0
)

print("\n" + "=" * 80)
print("修正统计:")
print(f"总错误样本数: {len(incorrect_samples)}")
print(f"修正成功数: {corrected_count}")
print(f"修正成功率: {success_rate * 100:.2f}%")


# =========================================================
# 保存
# =========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"cot_corrected_results_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(corrected_results, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存到: {output_file}")