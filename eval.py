from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ✅ 设置模型路径
MODEL_NAME = "./Qwen1.5-1.8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✅ 生成数学数据
def generate_math_data(num_samples=300):
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

synthetic_math_data = generate_math_data()
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

# ✅ 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 重新加载模型和 tokenizer
LOAD_PATH = "./ppo_trained_model"
trained_model = AutoModelForCausalLM.from_pretrained(LOAD_PATH, torch_dtype=torch.bfloat16).to(device)
trained_tokenizer = AutoTokenizer.from_pretrained(LOAD_PATH)

print("✅ PPO 训练后的模型已加载！")


def evaluate_model(model, tokenizer, dataset, num_samples=50):
    correct = 0
    total = min(num_samples, len(dataset))

    for sample in random.sample(list(dataset), total):
        question, correct_answer = sample["question"], sample["answer"]
        encoded_input = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        generated_output = model.generate(encoded_input.input_ids, max_length=50)
        generated_answer = tokenizer.decode(generated_output[0], skip_special_tokens=True)

        if correct_answer in generated_answer:
            correct += 1

    accuracy = correct / total
    return accuracy

# ✅ 评估 PPO 训练后的模型
trained_accuracy = evaluate_model(trained_model, trained_tokenizer, ds)
print(f"✅ PPO 训练后模型准确率: {trained_accuracy:.2%}")


# ✅ 加载原始模型
original_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)

test_questions = [
    "What is 15 + 27?",
    "What is 40 / 8?",
    "What is 6 * 9?",
    "What is 100 - 45?"
]

test_questions = [d["question"] for d in synthetic_math_data]

for question in test_questions:
    encoded_input = trained_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    # 原始模型预测
    original_output = original_model.generate(encoded_input.input_ids, max_length=50)
    original_answer = trained_tokenizer.decode(original_output[0], skip_special_tokens=True)

    # 训练后模型预测
    trained_output = trained_model.generate(encoded_input.input_ids, max_length=50)
    trained_answer = trained_tokenizer.decode(trained_output[0], skip_special_tokens=True)

    print(f"问题: {question}")
    print(f"原始模型: {original_answer}")
    print(f"训练后模型: {trained_answer}")
    print("-" * 50)
