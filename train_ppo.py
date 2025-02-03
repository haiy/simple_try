import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# trl 0.11.3

# ✅ 设置模型路径
MODEL_NAME = "./Qwen1.5-1.8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✅ 生成数学数据
def generate_math_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        a, b = random.randint(1, 100), random.randint(1, 100)
        operation = random.choice(["+", "-", "*", "/"])
        question = f"What is {a} {operation} {b}?"
        if operation == "/":
            answer = round(a / b, 2)  # 结果保留两位小数
        else:
            answer = eval(f"{a} {operation} {b}")
        data.append({"question": question, "answer": str(answer)})
    return data

synthetic_math_data = generate_math_data(500)
math_dataset = Dataset.from_dict({
    "question": [d["question"] for d in synthetic_math_data], 
    "answer": [d["answer"] for d in synthetic_math_data]
})

# ✅ 处理数据
def tokenize(sample):
    encoded_question = tokenizer(
        sample["question"], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )
    return {
        "input_ids": encoded_question["input_ids"].squeeze(0),
        "attention_mask": encoded_question["attention_mask"].squeeze(0),
        "answer": sample["answer"]
    }

ds = math_dataset.map(tokenize, batched=False)
ds.set_format(type="torch")

# ✅ 配置 PPO
config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1,
    log_with='wandb'  # 关闭 wandb 记录
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
model.gradient_checkpointing_enable()  # ✅ 减少显存占用

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=ds)

from torch.cuda.amp import autocast



# ✅ 训练循环
for epoch in range(3):  # 训练 3 个 epoch
    batch_size = 1  # 处理多个示例
    
    for batch in ds.shuffle().select(range(50)).train_test_split(test_size=0.2)["train"].batch(batch_size):  
        queries = batch["question"]  # 取多个 question
        correct_answers = batch["answer"]

        # ✅ 让模型生成答案
        encoded_queries = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        responses = model.generate(encoded_queries.input_ids, max_length=50)

        # ✅ 解码 batch 生成的答案
        decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)

        # ✅ 计算奖励（答案正确 +1，错误 -1）
        scores = list(torch.tensor(
            [1.0 if correct in response else -1.0 for correct, response in zip(correct_answers, decoded_responses)],
            dtype=torch.float,
            device=device
        ))

        # ✅ 确保 `queries` 和 `responses` 是 batch 维度对齐的张量
        queries_tensor = list(encoded_queries.input_ids)
        responses_tensor = list(tokenizer(decoded_responses, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to(device))

        # ✅ 进行 PPO 训练（batch 训练）
        ppo_trainer.step(queries_tensor, responses_tensor, scores)

    print(f"✅ Epoch {epoch + 1} 完成")

# ✅ 训练后评估
test_question = "What is 7 * 6?"
encoded_test_question = tokenizer(test_question, return_tensors="pt", padding=False, truncation=True, max_length=128).to(device)
result = model.generate(encoded_test_question.input_ids, max_length=50)
print("测试问题:", test_question)
print("模型答案:", tokenizer.decode(result[0], skip_special_tokens=True))
