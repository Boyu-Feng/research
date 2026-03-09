from datasets import load_dataset
from collections import defaultdict
import json
import re
import boto3

# ==============================
# AWS Bedrock Client
# ==============================

client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id="",
    aws_secret_access_key="",
    region_name="us-east-1"
)

# ==============================
# 调用 Bedrock 模型
# ==============================

def call_bedrock(prompt):

    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1024,
        "temperature": 1
    })

    response = client.invoke_model(
        modelId="openai.gpt-oss-safeguard-120b",
        body=body
    )

    result = json.loads(response["body"].read())

    print("Bedrock response:", result)

    return result["choices"][0]["message"]["content"]


# ==============================
# 提取答案
# ==============================

def extract_answer(text):
    """从答案文本中提取 ### 之后的数字"""
    if not text:
        return None

    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)

    if match:
        return match.group(1)

    return None


# ==============================
# 加载数据集
# ==============================

ds = load_dataset("apple/GSM-Symbolic", "p1")

print("="*80)
print("Dataset columns:", ds['test'].column_names)
print("Dataset size:", len(ds['test']))
print("\nFirst sample:")

for key in ds['test'].column_names:
    print(f'{key}: {ds["test"][0][key]}')

print("="*80)


# ==============================
# 按 original_id 分组
# ==============================

original_groups = defaultdict(list)

for item in ds['test']:
    original_id = item['original_id']
    original_groups[original_id].append(item)

print(f"\n总共有 {len(original_groups)} 个原始问题")


# ==============================
# 统计结果
# ==============================

results = {
    'total_original_questions': len(original_groups),
    'original_results': []
}


# ==============================
# 遍历每个原始问题
# ==============================

for idx, (original_id, items) in enumerate(original_groups.items()):

    if idx >= 100:
        break

    original_question = items[0]['original_question']
    original_answer = items[0]['original_answer']

    print(f"\n{'='*80}")
    print(f"原始问题 {idx+1}: {original_question}")
    print(f"原始答案: {original_answer}")

    # ==============================
    # 原问题推理
    # ==============================

    print("\n调用模型获取原始问题的答案...")

    try:

        original_model_answer = call_bedrock(
            original_question +
            "\n请在最后用 #### 格式输出答案，例如：#### 42"
        )

        print(f"模型答案: {original_model_answer[:300]}...")

        expected_num = extract_answer(original_answer)
        model_num = extract_answer(original_model_answer)

        original_is_correct = (
            expected_num is not None and model_num == expected_num
        )

        print(f"期望答案数字: {expected_num}, 模型答案数字: {model_num}")
        print(f"原问题是否正确: {'✓ 正确' if original_is_correct else '✗ 错误'}")

    except Exception as e:

        print(f"获取原始问题答案失败: {e}")

        original_model_answer = None
        original_is_correct = False


    # ==============================
    # 保存结果
    # ==============================

    original_result = {
        'original_id': original_id,
        'original_question': original_question,
        'original_answer': original_answer,
        'original_model_answer': original_model_answer,
        'original_is_correct': original_is_correct,
        'new_questions_results': []
    }

    print(f"\n处理该原始问题产生的 {len(items)} 个新问题...")

    correct_count = 0


    # ==============================
    # 新问题
    # ==============================

    for new_idx, item in enumerate(items):

        if new_idx >= 3:
            break

        new_question = item['question']
        expected_answer = item['answer']

        print(f"\n  新问题 {new_idx+1}: {new_question}")
        print(f"  期望答案: {expected_answer}")

        try:

            model_answer = call_bedrock(
                new_question +
                "\n请在最后用 #### 格式输出答案，例如：#### 42"
            )

            print(f"  模型答案: {model_answer[:300]}...")

            expected_num = extract_answer(expected_answer)
            model_num = extract_answer(model_answer)

            is_correct = (
                expected_num is not None and model_num == expected_num
            )

            print(f"  期望答案数字: {expected_num}, 模型答案数字: {model_num}")
            print(f"  是否正确: {'✓ 正确' if is_correct else '✗ 错误'}")

            if is_correct:
                correct_count += 1

            original_result['new_questions_results'].append({
                'question': new_question,
                'expected_answer': expected_answer,
                'model_answer': model_answer,
                'answer': model_num,
                'is_correct': is_correct
            })

        except Exception as e:
            print(f"  获取答案失败: {e}")


    accuracy = correct_count / min(3, len(items)) if items else 0

    original_result['new_questions_accuracy'] = accuracy

    results['original_results'].append(original_result)

    print(f"\n该原始问题的新问题准确率: {accuracy*100:.1f}%")


# ==============================
# 汇总
# ==============================

print(f"\n\n{'='*80}")
print("汇总结果:")
print(f"总原始问题数: {results['total_original_questions']}")
print(f"处理的原始问题数: {len(results['original_results'])}")


for idx, original_result in enumerate(results['original_results']):

    print(f"\n原始问题 {idx+1}: {original_result['original_question']}")
    print(f"  原问题是否正确: {'✓ 正确' if original_result['original_is_correct'] else '✗ 错误'}")
    print(f"  新问题准确率: {original_result['new_questions_accuracy']*100:.1f}%")
    print(f"  处理的新问题数: {len(original_result['new_questions_results'])}")


# ==============================
# 保存结果
# ==============================

with open('gsm_symbolic_results.json', 'w', encoding='utf-8') as f:

    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n详细结果已保存到 gsm_symbolic_results.json")