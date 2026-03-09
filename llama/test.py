from typing import List, Dict, Tuple
from datasets import load_dataset
from collections import defaultdict
import json
import re
import fire
import numpy as np
import torch

from generation import Llama


def build_prompt(question):
    """构建提示词"""
    prompt = f"""Please reason step by step about the following problem.

Problem: {question}

Think carefully step by step:

Please output the final answer finally after ###.

For example ### 42

"""
    return prompt


def extract_answer(text):
    """从答案文本中提取 ### 之后的数字"""
    if not text:
        return None
    match = re.search(r'###\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    return None


def generate_attention_heatmap_html(
    tokens: List[str],
    attention_weights: np.ndarray,
    prompt_length: int,
    question_text: str,
    answer_text: str,
    title: str = "Attention Visualization"
):
    """
    生成注意力热力图的HTML可视化
    
    Args:
        tokens: token列表
        attention_weights: 注意力权重矩阵 [seq_len, seq_len]
        prompt_length: prompt的token数量
        question_text: 问题文本
        answer_text: 答案文本
        title: 图表标题
    """
    
    # 限制显示的token数量（避免HTML太大）
    max_display_tokens = 100
    if len(tokens) > max_display_tokens:
        # 只显示前面的prompt和最后的生成部分
        display_tokens = tokens[:prompt_length] + tokens[-50:]
        display_attention = np.concatenate([
            attention_weights[:prompt_length, :],
            attention_weights[-50:, :]
        ], axis=0)
        display_attention = np.concatenate([
            display_attention[:, :prompt_length],
            display_attention[:, -50:]
        ], axis=1)
    else:
        display_tokens = tokens
        display_attention = attention_weights
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .question-box, .answer-box {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #4CAF50;
        }}
        .answer-box {{
            border-left-color: #2196F3;
        }}
        .question-box h3, .answer-box h3 {{
            margin-top: 0;
            color: #555;
        }}
        .heatmap-container {{
            overflow-x: auto;
            margin: 20px 0;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
        }}
        .heatmap {{
            display: grid;
            gap: 1px;
            background-color: #ddd;
            margin: 20px 0;
        }}
        .heatmap-cell {{
            min-width: 30px;
            min-height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            transition: transform 0.2s;
            cursor: pointer;
        }}
        .heatmap-cell:hover {{
            transform: scale(1.1);
            z-index: 10;
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
        }}
        .token-label {{
            background-color: #f0f0f0;
            font-weight: bold;
            padding: 5px;
            font-size: 11px;
            word-break: break-all;
        }}
        .prompt-token {{
            background-color: #e8f5e9;
        }}
        .generation-token {{
            background-color: #e3f2fd;
        }}
        .legend {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-color {{
            width: 30px;
            height: 20px;
            border-radius: 3px;
        }}
        .gradient-bar {{
            width: 300px;
            height: 20px;
            background: linear-gradient(to right, #f7fbff, #08306b);
            border-radius: 3px;
            margin: 0 10px;
        }}
        .info {{
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #9C27B0;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .stat-value {{
            color: #333;
            font-size: 24px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="question-box">
            <h3>问题:</h3>
            <p>{question_text}</p>
        </div>
        
        <div class="answer-box">
            <h3>模型回答:</h3>
            <p>{answer_text[:500]}{'...' if len(answer_text) > 500 else ''}</p>
        </div>
        
        <div class="info">
            <strong>说明:</strong> 热力图展示了每个生成token对之前所有token的注意力分布。
            颜色越深表示注意力权重越高。绿色背景是prompt部分，蓝色背景是生成部分。
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Prompt Token数</div>
                <div class="stat-value">{prompt_length}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">生成Token数</div>
                <div class="stat-value">{len(tokens) - prompt_length}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">总Token数</div>
                <div class="stat-value">{len(tokens)}</div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color prompt-token"></div>
                <span>Prompt Token</span>
            </div>
            <div class="legend-item">
                <div class="legend-color generation-token"></div>
                <span>生成Token</span>
            </div>
            <div class="legend-item">
                <span>注意力强度:</span>
                <div class="gradient-bar"></div>
                <span>低 → 高</span>
            </div>
        </div>
        
        <div class="heatmap-container">
            <div class="heatmap" style="grid-template-columns: 150px repeat({len(display_tokens)}, 30px);">
"""
    
    # 添加列标签（横向token）
    html += '<div class="token-label">To Token →</div>'
    for i, token in enumerate(display_tokens):
        token_class = 'prompt-token' if i < prompt_length else 'generation-token'
        # 清理token显示
        display_token = token.replace('<', '&lt;').replace('>', '&gt;').replace(' ', '␣')
        html += f'<div class="token-label {token_class}" title="{display_token}">{display_token[:8]}</div>'
    
    # 添加热力图行
    for i, token in enumerate(display_tokens):
        # 行标签（纵向token）
        token_class = 'prompt-token' if i < prompt_length else 'generation-token'
        display_token = token.replace('<', '&lt;').replace('>', '&gt;').replace(' ', '␣')
        html += f'<div class="token-label {token_class}" title="{display_token}">{display_token[:20]}</div>'
        
        # 注意力权重单元格
        for j in range(len(display_tokens)):
            if j <= i:  # 只显示因果注意力（当前token只能看到之前的token）
                weight = display_attention[i, j]
                # 使用蓝色渐变
                color_intensity = int(weight * 255)
                color = f'rgb({255-color_intensity}, {255-color_intensity}, 255)'
                html += f'<div class="heatmap-cell" style="background-color: {color};" title="从 {i} 到 {j}: {weight:.4f}"></div>'
            else:
                html += '<div class="heatmap-cell" style="background-color: #f0f0f0;"></div>'
    
    html += """
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return html


def analyze_attention_patterns(
    attention_weights: np.ndarray,
    prompt_length: int,
    tokens: List[str]
) -> Dict:
    """
    分析注意力模式的统计信息
    
    Returns:
        包含各种统计指标的字典
    """
    seq_len = len(tokens)
    gen_length = seq_len - prompt_length
    
    stats = {
        'prompt_length': prompt_length,
        'generation_length': gen_length,
        'total_length': seq_len,
        'avg_attention_to_prompt': [],  # 每个生成token对prompt的平均注意力
        'avg_attention_to_generation': [],  # 每个生成token对已生成部分的平均注意力
        'max_attention_positions': [],  # 每个token最关注的位置
    }
    
    # 分析生成部分的注意力模式
    for i in range(prompt_length, seq_len):
        # 对prompt的平均注意力
        prompt_attention = np.mean(attention_weights[i, :prompt_length])
        stats['avg_attention_to_prompt'].append(prompt_attention)
        
        # 对已生成部分的平均注意力
        if i > prompt_length:
            gen_attention = np.mean(attention_weights[i, prompt_length:i])
            stats['avg_attention_to_generation'].append(gen_attention)
        else:
            stats['avg_attention_to_generation'].append(0.0)
        
        # 最大注意力位置
        max_pos = np.argmax(attention_weights[i, :i+1])
        stats['max_attention_positions'].append(max_pos)
    
    return stats


def compare_attention_patterns_html(
    original_stats: Dict,
    new_stats_list: List[Dict],
    original_question: str,
    new_questions: List[str]
):
    """生成对比多个问题注意力模式的HTML"""
    
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>注意力模式对比</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .chart-wrapper {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }
        .question-box {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #4CAF50;
        }
        .new-question-box {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #2196F3;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-label {
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .stat-value {
            color: #333;
            font-size: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>原始问题 vs 新问题 - 注意力模式对比</h1>
        
        <div class="question-box">
            <h3>原始问题:</h3>
            <p>{}</p>
        </div>
""".format(original_question)
    
    for i, q in enumerate(new_questions):
        html += f"""
        <div class="new-question-box">
            <h3>新问题 {i+1}:</h3>
            <p>{q}</p>
        </div>
"""
    
    # 统计对比
    html += """
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">原始问题 Prompt长度</div>
                <div class="stat-value">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">原始问题 生成长度</div>
                <div class="stat-value">{}</div>
            </div>
""".format(original_stats['prompt_length'], original_stats['generation_length'])
    
    for i, stats in enumerate(new_stats_list):
        html += f"""
            <div class="stat-card">
                <div class="stat-label">新问题{i+1} Prompt长度</div>
                <div class="stat-value">{stats['prompt_length']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">新问题{i+1} 生成长度</div>
                <div class="stat-value">{stats['generation_length']}</div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="chart-container">
            <h2>对Prompt的平均注意力（按生成位置）</h2>
            <div class="chart-wrapper">
                <canvas id="promptAttentionChart"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>对已生成部分的平均注意力（按生成位置）</h2>
            <div class="chart-wrapper">
                <canvas id="generationAttentionChart"></canvas>
            </div>
        </div>
        
        <script>
"""
    
    # 准备图表数据
    html += f"""
        const originalPromptAttention = {json.dumps(original_stats['avg_attention_to_prompt'])};
        const originalGenAttention = {json.dumps(original_stats['avg_attention_to_generation'])};
"""
    
    for i, stats in enumerate(new_stats_list):
        html += f"""
        const new{i}PromptAttention = {json.dumps(stats['avg_attention_to_prompt'])};
        const new{i}GenAttention = {json.dumps(stats['avg_attention_to_generation'])};
"""
    
    # Prompt注意力图表
    html += """
        const promptCtx = document.getElementById('promptAttentionChart').getContext('2d');
        new Chart(promptCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: originalPromptAttention.length}, (_, i) => i),
                datasets: [
                    {
                        label: '原始问题',
                        data: originalPromptAttention,
                        borderColor: 'rgb(76, 175, 80)',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.1
                    },
"""
    
    colors = [
        'rgb(33, 150, 243)',
        'rgb(255, 152, 0)',
        'rgb(156, 39, 176)',
        'rgb(244, 67, 54)',
    ]
    
    for i in range(len(new_stats_list)):
        html += f"""
                    {{
                        label: '新问题{i+1}',
                        data: new{i}PromptAttention,
                        borderColor: '{colors[i % len(colors)]}',
                        backgroundColor: '{colors[i % len(colors)].replace("rgb", "rgba").replace(")", ", 0.1)")}',
                        tension: 0.1
                    }},
"""
    
    html += """
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '平均注意力权重'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '生成位置（相对于prompt结束）'
                        }
                    }
                }
            }
        });
"""
    
    # 生成部分注意力图表
    html += """
        const genCtx = document.getElementById('generationAttentionChart').getContext('2d');
        new Chart(genCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: originalGenAttention.length}, (_, i) => i),
                datasets: [
                    {
                        label: '原始问题',
                        data: originalGenAttention,
                        borderColor: 'rgb(76, 175, 80)',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.1
                    },
"""
    
    for i in range(len(new_stats_list)):
        html += f"""
                    {{
                        label: '新问题{i+1}',
                        data: new{i}GenAttention,
                        borderColor: '{colors[i % len(colors)]}',
                        backgroundColor: '{colors[i % len(colors)].replace("rgb", "rgba").replace(")", ", 0.1)")}',
                        tension: 0.1
                    }},
"""
    
    html += """
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '平均注意力权重'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '生成位置（相对于prompt结束）'
                        }
                    }
                }
            }
        });
        </script>
    </div>
</body>
</html>
"""
    
    return html


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 512,
    max_batch_size: int = 1,  # 设为1以便获取注意力权重
    max_original_questions: int = 5,
    max_new_questions_per_original: int = 3,
    output_dir: str = "attention_analysis",
):
    """
    分析模型在GSM-Symbolic数据集上的注意力模式
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
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

    # 遍历原始问题
    for idx, (original_id, items) in enumerate(origin_groups.items()):
        if idx >= max_original_questions:  
            break
    
        if not items:
            continue
            
        original_question = items[0]['original_question']
        original_answer = items[0]['original_answer']
        
        print(f"\n{'='*80}")
        print(f"分析原始问题 {idx+1}: {original_question[:100]}...")

        # 生成原始问题的回答并获取注意力
        prompt = build_prompt(original_question)
        
        try:
            # 注意：这里需要修改Llama类以返回注意力权重
            # 假设返回格式包含 'generation', 'tokens', 'attention_weights'
            result = generator.text_completion(
                [prompt],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            
            # 提取结果（需要根据实际的Llama实现调整）
            model_answer = result[0]['generation']
            
            # 如果Llama类返回了tokens和attention
            if 'tokens' in result[0] and 'attention_weights' in result[0]:
                tokens = result[0]['tokens']
                attention_weights = result[0]['attention_weights']  # shape: [seq_len, seq_len]
                
                # 计算prompt长度
                prompt_tokens = generator.tokenizer.encode(prompt, bos=True, eos=False)
                prompt_length = len(prompt_tokens)
                
                # 生成热力图
                html = generate_attention_heatmap_html(
                    tokens=tokens,
                    attention_weights=attention_weights,
                    prompt_length=prompt_length,
                    question_text=original_question,
                    answer_text=model_answer,
                    title=f"原始问题 {idx+1} - 注意力可视化"
                )
                
                output_file = os.path.join(output_dir, f"original_{idx+1}_attention.html")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"已保存原始问题注意力可视化: {output_file}")
                
                # 分析注意力模式
                original_stats = analyze_attention_patterns(
                    attention_weights, prompt_length, tokens
                )
                
                # 处理新问题
                new_stats_list = []
                new_questions = []
                
                for new_idx, new_item in enumerate(items[:max_new_questions_per_original]):
                    new_question = new_item['question']
                    new_questions.append(new_question)
                    
                    print(f"  分析新问题 {new_idx+1}...")
                    new_prompt = build_prompt(new_question)
                    
                    new_result = generator.text_completion(
                        [new_prompt],
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    
                    if 'tokens' in new_result[0] and 'attention_weights' in new_result[0]:
                        new_tokens = new_result[0]['tokens']
                        new_attention = new_result[0]['attention_weights']
                        new_answer = new_result[0]['generation']
                        
                        new_prompt_tokens = generator.tokenizer.encode(new_prompt, bos=True, eos=False)
                        new_prompt_length = len(new_prompt_tokens)
                        
                        # 生成热力图
                        new_html = generate_attention_heatmap_html(
                            tokens=new_tokens,
                            attention_weights=new_attention,
                            prompt_length=new_prompt_length,
                            question_text=new_question,
                            answer_text=new_answer,
                            title=f"新问题 {idx+1}-{new_idx+1} - 注意力可视化"
                        )
                        
                        new_output_file = os.path.join(
                            output_dir, 
                            f"original_{idx+1}_new_{new_idx+1}_attention.html"
                        )
                        with open(new_output_file, 'w', encoding='utf-8') as f:
                            f.write(new_html)
                        print(f"  已保存新问题注意力可视化: {new_output_file}")
                        
                        # 分析注意力模式
                        new_stats = analyze_attention_patterns(
                            new_attention, new_prompt_length, new_tokens
                        )
                        new_stats_list.append(new_stats)
                
                # 生成对比图表
                if new_stats_list:
                    comparison_html = compare_attention_patterns_html(
                        original_stats=original_stats,
                        new_stats_list=new_stats_list,
                        original_question=original_question,
                        new_questions=new_questions
                    )
                    
                    comparison_file = os.path.join(
                        output_dir, 
                        f"original_{idx+1}_comparison.html"
                    )
                    with open(comparison_file, 'w', encoding='utf-8') as f:
                        f.write(comparison_html)
                    print(f"已保存对比分析: {comparison_file}")
            
            else:
                print("警告: 模型未返回attention weights，请修改Llama类以支持返回注意力权重")
                
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"分析完成！所有可视化文件已保存到: {output_dir}")
    print(f"请用浏览器打开HTML文件查看注意力可视化")


if __name__ == "__main__":
    fire.Fire(main(ckpt_dir='llama\checkpoints\original' , tokenizer_path=r"llama\checkpoints\original\tokenizer.model"))
