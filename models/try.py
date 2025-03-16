from models.tinyllm_dataset import SFTDataset, RLDataset, load_ppo_dataset, load_dpo_dataset
import transformers

data_path = "/home/wenjinyong/DATAS/tinyllm_dataset/sft_train/sft_data.jsonl"
base_model_path = "tiny_llm_sft_92m"
model_max_length=512

tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=model_max_length
    )

# sft_dataset = SFTDataset(
#         data_path, 
#         tokenizer, 
#         model_max_length
#     )
# print(sft_dataset)

# print(sft_dataset[0])
# print(len(sft_dataset))
# print(sft_dataset[len(sft_dataset)])

data_path = "/home/wenjinyong/DATAS/tinyllm_dataset/rl_train/rl_train_data.jsonl"

rl_dataset = RLDataset(
        data_path, 
        tokenizer, 
        model_max_length
    )
print(rl_dataset[0])

rl_ds = load_ppo_dataset(data_path, tokenizer, max_length=512, sanity_check=True)
print(rl_ds[0])
rl_ds = load_dpo_dataset(data_path, max_length=512, sanity_check=True)
print(rl_ds[0])


