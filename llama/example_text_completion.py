from typing import List
from datasets import load_dataset
from collections import defaultdict
import json
import re
import fire
import os
from datetime import datetime

from generation import Llama


def build_prompt(question):
    """构建提示词"""
    prompt = f"""Please reason step by step about the following problem.

Problem: {question}

Think carefully step by step:

Please output the final answer finally after ####.

For example #### 42

"""
    return prompt


def extract_answer(text):
    """从答案文本中提取 #### 之后的数字"""
    if not text:
        return None
    # 查找 #### 并提取之后的数字
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    return None


def save_incremental_result(result_item, output_file):
    """增量保存单条结果到JSON文件"""
    # 读取现有结果
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {'results': []}
    else:
        data = {'results': []}
    
    # 添加新结果
    data['results'].append(result_item)
    
    # 保存回文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calculate_statistics(results_file):
    """计算最终统计数据"""
    if not os.path.exists(results_file):
        print(f"结果文件不存在: {results_file}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    if not results:
        print("没有找到任何结果")
        return
    
    # 统计原始问题
    total_original = len(results)
    original_correct = sum(1 for r in results if r.get('original_is_correct', False))
    original_accuracy = original_correct / total_original if total_original > 0 else 0
    
    # 统计新问题
    total_new_questions = 0
    new_questions_correct = 0
    
    for result in results:
        new_results = result.get('new_questions_results', [])
        total_new_questions += len(new_results)
        new_questions_correct += sum(1 for nr in new_results if nr.get('is_correct', False))
    
    new_questions_accuracy = new_questions_correct / total_new_questions if total_new_questions > 0 else 0
    
    # 准备统计信息
    statistics = {
        'timestamp': datetime.now().isoformat(),
        'original_questions': {
            'total': total_original,
            'correct': original_correct,
            'accuracy': original_accuracy,
            'accuracy_percent': f"{original_accuracy * 100:.2f}%"
        },
        'new_questions': {
            'total': total_new_questions,
            'correct': new_questions_correct,
            'accuracy': new_questions_accuracy,
            'accuracy_percent': f"{new_questions_accuracy * 100:.2f}%"
        },
        'performance_drop': {
            'absolute': original_accuracy - new_questions_accuracy,
            'relative_percent': f"{((original_accuracy - new_questions_accuracy) / original_accuracy * 100):.2f}%" if original_accuracy > 0 else "N/A"
        }
    }
    
    # 将统计信息添加到数据中
    data['statistics'] = statistics
    
    # 保存更新后的数据
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return statistics


def main(
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 16000,
    max_gen_len: int = 16000,
    max_batch_size: int = 4,
    max_original_questions: int = 100,
    max_new_questions_per_original: int = 3,
    output_file: str = None,
):
    """
    在 GSM-Symbolic 数据集上评估模型性能
    
    Args:
        temperature: 生成温度
        top_p: nucleus sampling 参数
        max_seq_len: 最大序列长度
        max_gen_len: 最大生成长度
        max_batch_size: 最大批次大小
        max_original_questions: 最多评估多少个原始问题
        max_new_questions_per_original: 每个原始问题最多评估多少个新问题
        output_file: 输出文件路径（默认为 test_results_TIMESTAMP.json）
    """
    # 硬编码权重路径
    ckpt_dir = 'checkpoints/Meta-Llama-3.1-8B-Instruct/original'
    tokenizer_path = 'checkpoints/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model'
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'test_results_{timestamp}.json'
    
    print(f"使用模型路径: {ckpt_dir}")
    print(f"使用tokenizer路径: {tokenizer_path}")
    print(f"结果将保存到: {output_file}")
    
    # 如果文件已存在，先删除（重新开始）
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除旧的结果文件")
    
    # 初始化模型
    print("正在加载模型...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("模型加载完成!")
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset("apple/GSM-Symbolic", "p1")
    origin_groups = defaultdict(list)

    for item in dataset['test']:
        original_id = item['original_id']
        origin_groups[original_id].append(item)

    print(f"总原始问题数: {len(origin_groups)}")
    if origin_groups:
        sample_items = list(origin_groups.values())[0]
        print(f"每个原始问题的新问题数示例: {len(sample_items)}")

    # 遍历原始问题
    processed_count = 0
    
    for idx, (original_id, items) in enumerate(origin_groups.items()):
        if idx >= max_original_questions:  
            break
    
        if not items:
            continue
            
        original_question = items[0]['original_question']
        original_answer = items[0]['original_answer']
        
        print(f"\n{'='*80}")
        print(f"原始问题 {idx+1}/{min(max_original_questions, len(origin_groups))}: {original_question}")
        print(f"原始答案: {original_answer}")

        # 初始化当前原始问题的结果
        original_result = {
            'index': idx + 1,
            'original_id': original_id,
            'original_question': original_question,
            'original_answer': original_answer,
            'original_model_answer': None,
            'original_model_extracted': None,
            'original_is_correct': False,
            'new_questions_results': [],
        }

        # 评估原始问题
        prompt = build_prompt(original_question)
        try:
            model_results = generator.text_completion(
                [prompt],  # 传入列表
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            
            # 提取模型回答
            model_answer = model_results[0]['generation'] if model_results else ""
            original_result['original_model_answer'] = model_answer
            
            # 比较答案
            expected_num = extract_answer(original_answer)
            model_num = extract_answer(model_answer)
            original_result['original_model_extracted'] = model_num
            answer_is_correct = (
                expected_num is not None 
                and model_num is not None 
                and float(expected_num) == float(model_num)
            )
            original_result['original_is_correct'] = answer_is_correct
            
            print(f"期望答案数字: {expected_num}, 模型答案数字: {model_num}")
            print(f"原问题是否正确: {'✓ 正确' if answer_is_correct else '✗ 错误'}")
            print(f"模型完整回答:\n{model_answer[:200]}...")
            
        except Exception as e:
            print(f"获取原始问题答案失败: {e}")
            original_result['original_error'] = str(e)

        # 评估新问题
        print(f"\n评估该原始问题的新问题变体...")
        correct_count = 0
        
        # 限制评估的新问题数量
        new_questions_to_evaluate = items[:max_new_questions_per_original]
        
        for new_idx, new_item in enumerate(new_questions_to_evaluate):
            new_question = new_item['question']
            expected_answer = new_item['answer']
            
            print(f"\n  新问题 {new_idx+1}/{len(new_questions_to_evaluate)}: {new_question[:100]}...")
            
            new_prompt = build_prompt(new_question)
            new_question_result = {
                'index': new_idx + 1,
                'question': new_question,
                'expected_answer': expected_answer,
                'model_answer': None,
                'model_extracted': None,
                'is_correct': False
            }
            
            try:
                new_model_results = generator.text_completion(
                    [new_prompt],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                
                model_answer = new_model_results[0]['generation'] if new_model_results else ""
                new_question_result['model_answer'] = model_answer
                
                expected_num = extract_answer(expected_answer)
                model_num = extract_answer(model_answer)
                new_question_result['model_extracted'] = model_num
                
                is_correct = (
                    expected_num is not None 
                    and model_num is not None 
                    and float(expected_num) == float(model_num)
                )

                new_question_result['is_correct'] = is_correct
                
                if is_correct:
                    correct_count += 1
                
                print(f"  期望: {expected_num}, 模型: {model_num}, {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                print(f"  评估新问题失败: {e}")
                new_question_result['error'] = str(e)
            
            original_result['new_questions_results'].append(new_question_result)
        
        # 计算新问题准确率
        accuracy = correct_count / len(new_questions_to_evaluate) if new_questions_to_evaluate else 0
        original_result['new_questions_accuracy'] = accuracy
        original_result['new_questions_correct_count'] = correct_count
        original_result['new_questions_total_count'] = len(new_questions_to_evaluate)
        
        print(f"\n该原始问题的新问题准确率: {accuracy*100:.1f}% ({correct_count}/{len(new_questions_to_evaluate)})")
        
        # 保存当前结果
        save_incremental_result(original_result, output_file)
        processed_count += 1
        print(f"已保存第 {processed_count} 条结果到 {output_file}")

    # 计算并保存最终统计
    print(f"\n{'='*80}")
    print("正在计算最终统计数据...")
    statistics = calculate_statistics(output_file)
    
    if statistics:
        print(f"\n{'='*80}")
        print("最终统计结果:")
        print(f"\n原始问题:")
        print(f"  总数: {statistics['original_questions']['total']}")
        print(f"  正确: {statistics['original_questions']['correct']}")
        print(f"  准确率: {statistics['original_questions']['accuracy_percent']}")
        
        print(f"\n新问题:")
        print(f"  总数: {statistics['new_questions']['total']}")
        print(f"  正确: {statistics['new_questions']['correct']}")
        print(f"  准确率: {statistics['new_questions']['accuracy_percent']}")
        
        print(f"\n性能下降:")
        print(f"  绝对下降: {statistics['performance_drop']['absolute']:.4f}")
        print(f"  相对下降: {statistics['performance_drop']['relative_percent']}")
    
    print(f"\n{'='*80}")
    print(f"所有结果已保存到: {output_file}")
    print(f"处理了 {processed_count} 个原始问题")


if __name__ == "__main__":
    fire.Fire(main)