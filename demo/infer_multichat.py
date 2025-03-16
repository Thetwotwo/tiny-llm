from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import torch

# 加载模型和 Tokenizer
model_id = "../tiny_llm_sft_92m"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, local_files_only=True)
generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)

generation_config.max_new_tokens = 200

# 记录对话历史
chat_history = []

def chat(user_text):
    global chat_history
    
    # 组织输入，包含历史对话
    # input_txt = "\n".join(["<|system|> 你是一个智能助手，帮助用户回答问题。"] + chat_history + [f"<|user|> {user_text}", "<|assistant|>"]).strip() + "\n"
    input_txt = "\n".join(["<|system|> 你是一个智能助手，帮助用户回答问题。"] + chat_history + [f"<|user|>\n{user_text}", "<|assistant|>"]).strip() + "\n"

    model_inputs = tokenizer(input_txt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, generation_config=generation_config)
    generated_ids = [
        output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 更新对话历史
    # chat_history.append(f"<|user|> {user_text}")
    # chat_history.append(f"<|assistant|> {response}")
    chat_history.append(f"<|user|>\n{user_text}")
    chat_history.append(f"<|assistant|>\n{response}")    
    return response

# 示例对话
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Assistant: 再见！")
        break
    response = chat(user_input)
    print(f"Assistant: {response}")