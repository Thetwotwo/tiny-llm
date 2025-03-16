from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation import GenerationConfig
import os

# model_id = "outputs/ckpt/tiny_llm_sft_92m"
model_id = "wdndev/tiny_llm_sft_92m"
model_id = "outputs/tiny_llm_sft_76m_llama"
model_id = "../tiny_llm_sft_92m"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
quantization_config = BitsAndBytesConfig(load_in_8bit=False)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",  trust_remote_code=True, local_files_only=True, quantization_config=quantization_config)
generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)


sys_text = "你是一个智能助手，帮助用户回答问题。"
# user_text = "世界上最大的动物是什么？"
# user_text = "介绍一下刘德华。"
user_text = "介绍一下中国。"
input_txt = "\n".join(["<|system|>", sys_text.strip(), 
                        "<|user|>", user_text.strip(), 
                        "<|assistant|>"]).strip() + "\n"

generation_config.max_new_tokens = 200
model_inputs = tokenizer(input_txt, return_tensors="pt").to(model.device)

print(model_inputs)

generated_ids = model.generate(model_inputs.input_ids, generation_config=generation_config)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


