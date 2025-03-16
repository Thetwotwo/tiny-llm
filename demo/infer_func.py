import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

def load_model_tokenizer(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(model_id)
    return model, tokenizer, generation_config

def tinyllm_infer(text: str, 
                  model: AutoModelForCausalLM, 
                  tokenizer: AutoTokenizer, 
                  generation_config: GenerationConfig
    ):
    sys_text = "你是个人助手。"
    input_txt = "\n".join(["<|system|>", sys_text.strip(), 
                        "<|user|>", text.strip(), 
                        "<|assistant|>"]).strip() + "\n"
    model_inputs = tokenizer(input_txt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, generation_config=generation_config)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def main():
    model_id = "../tiny_llm_sft_92m"

    model, tokenizer, generation_config = load_model_tokenizer(model_id)
    generation_config.max_new_tokens = 2000
    input_text = "介绍一下中国"
    response = tinyllm_infer(input_text, model, tokenizer, generation_config)

    print(response)

if __name__ == "__main__":
    main()