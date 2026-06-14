import json
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from PIL import Image
import os

def evaluate_model(model_path, test_data_path, output_dir, device="cuda"):
    """评估模型在测试集上的表现"""
    
    print(f"加载模型: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = Qwen3VLProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载测试数据
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    results = []
    correct_count = 0
    
    for i, item in enumerate(test_data):
        print(f"\n{'='*60}")
        print(f"问题 {i+1}/{len(test_data)}: {item['id']}")
        
        # 构建消息
        messages = []
        images = []
        
        for msg in item["messages"]:
            if msg["from"] == "system":
                messages.append({"role": "system", "content": msg["value"]})
            elif msg["from"] == "human":
                content = msg["value"]
                # 处理 <image> 标记
                if "<image>" in content:
                    content = content.replace("<image>", "")
                    if item.get("images"):
                        img_path = item["images"][0]
                        # 处理相对路径
                        if img_path.startswith("data/"):
                            img_path = os.path.join(os.path.dirname(test_data_path), "..", img_path)
                        images.append(Image.open(img_path).convert("RGB"))
                messages.append({"role": "user", "content": content.strip()})
        
        # 推理
        try:
            if images:
                inputs = processor(
                    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    images=images,
                    return_tensors="pt"
                ).to(model.device)
            else:
                inputs = processor(
                    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    return_tensors="pt"
                ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )
            
            predicted = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            # 提取 assistant 部分
            if "assistant" in predicted:
                predicted = predicted.split("assistant")[-1].strip()
            
        except Exception as e:
            print(f"推理错误: {e}")
            predicted = f"ERROR: {str(e)}"
        
        # 获取正确答案
        answer_msg = [m for m in item["messages"] if m["from"] == "gpt"][0]
        expected = answer_msg["value"]
        
        # 简单准确率判断（答案包含关系）
        is_correct = False
        if "ERROR" not in predicted:
            # 提取关键信息判断
            if "部件4" in expected and "配气带孔盘" in predicted:
                is_correct = True
            elif "P·吉特勒" in expected and "P·吉特勒" in predicted:
                is_correct = True
            elif "含尘气体" in expected and "含尘气体" in predicted:
                is_correct = True
            elif "h1" in expected and "h2" in expected and "位置" in predicted:
                is_correct = True
            elif "6''" in expected and "6''" in predicted:
                is_correct = True
            elif "X1" in expected and "间隔" in predicted:
                is_correct = True
            elif "圆锥形" in expected and "圆锥形" in predicted:
                is_correct = True
            elif "10" in expected and "主体框架" in predicted:
                is_correct = True
            elif "14 → 13 → 12" in expected and ("14" in predicted and "13" in predicted):
                is_correct = True
            elif "部件13" in expected and "上方" in predicted:
                is_correct = True
        
        if is_correct:
            correct_count += 1
        
        results.append({
            "id": item["id"],
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct
        })
        
        print(f"预测: {predicted[:200]}...")
        print(f"正确: {is_correct}")
    
    # 计算准确率
    accuracy = correct_count / len(test_data) * 100
    
    print(f"\n{'='*60}")
    print(f"评估完成!")
    print(f"总问题数: {len(test_data)}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy:.1f}%")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "model": model_path,
            "total": len(test_data),
            "correct": correct_count,
            "accuracy": accuracy,
            "details": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_dir}/results.json")
    
    return accuracy

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("用法: python3 evaluate_model.py <model_path> <test_data> <output_dir>")
        print("示例:")
        print("  python3 evaluate_model.py models/Qwen3-VL-Industrial-Finetuned data/test_set_10_sharegpt.jsonl eval_finetuned")
        print("  python3 evaluate_model.py /home/ztt/models/Qwen3-VL-2B-Instruct data/test_set_10_sharegpt.jsonl eval_baseline")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data = sys.argv[2]
    output_dir = sys.argv[3]
    
    evaluate_model(model_path, test_data, output_dir)
