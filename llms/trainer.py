from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
    AutoPeftModelForCausalLM
)
import os, torch, wandb, fire, evaluate
from loguru import logger
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()

hf_token = user_secrets.get_secret("Hugging-face")

login(token = hf_token)

wb_token = user_secrets.get_secret("wandb")

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3 8B on Medical Dataset',
    job_type="training",
    anonymous="allow"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics = evaluate.load('rouge')

class FineTuneLLMs:
    def __init__(self,
                 pretrained_model_name_or_path: str = 'SeaLLMs/SeaLLMs-v3-1.5B',
                 checkpoint_path: str = './cp',
                 max_len: int = 2048,
                 seed: int = 42,
                 num_proc: int = 16,
                 data_path: str = 'shnl/ViCivicEdu',
                 per_device_train_batch_size: int = 4,
                 per_device_eval_batch_size: int = 8,
                 num_epochs: int = 1,
                 lr: float = 1e-4,
                 warmup_ratio: float = 0.05,
                 weight_decay: float = 0.015,
                 use_flash_attention: bool = False,
                 type_qg: str = 'qag'):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.checkpoint_path = checkpoint_path
        self.max_len = max_len
        self.seed = seed
        self.num_proc = num_proc
        self.data_path = data_path
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.use_flash_attention = use_flash_attention
        self.type = type_qg
    def prepare_data(self):
        logger.info('-----:----- Preparing dataset -----:-----')
        dataset = load_dataset(f'{self.data_path}', use_auth_token=True)
        train_dataset = dataset['train']
        dev_dataset = dataset['validation']
        test_dataset = dataset['test']
        return train_dataset, dev_dataset, test_dataset
    def train(self):
        print(
            f"Training AlphaEdu llms model with params:\n"
            f"model: {self.pretrained_model_name_or_path}\n"
            f"checkpoint: {self.checkpoint_path}\n"
            f"max_len: {self.max_len}\n"
            f"seed: {self.seed}\n"
            f"num_proc: {self.num_proc}\n"
            f"data_path: {self.data_path}\n"
            f"per_device_train_batch_size: {self.per_device_train_batch_size}\n"
            f"per_device_eval_batch_size: {self.per_device_eval_batch_size}\n"
            f"num_epochs: {self.num_epochs}\n"
            f"lr: {self.lr}\n"
            f"warmup_ratio: {self.warmup_ratio}\n"
            f"weight_decay: {self.weight_decay}\n"
            f"use_flash_attention: {self.use_flash_attention}\n"
            f"type: {self.type}\n"
        )
        assert (
            self.pretrained_model_name_or_path
        ), "Please specify a --base_model, e.g. --base_model='SeaLLMs/SeaLLMs-v3-1.5B'"

        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager"
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # LoRA config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        train_dataset, dev_dataset, _ = self.prepare_data()
        def formatting_func_qag(example):
            return (
                "<|im_start|> user \n"
                f"{example['instruction']} "
                f"Context: {example['paragraph']} <|im_end|> \n"
                f"<|im_start|>assistant {example['questions_answers']}"
            )
        def formatting_func_mcq(example):
            return (
                "<|im_start|> user \n"
                f"{example['paragraph']} {example['answer']} {example['question']} <|im_end|> \n"
                f"<|im_start|>assistant {example['distract']}"
            )
        def formatting_func_gapfill(example):
            return (
                "<|im_start|> user \n"
                f"{example['paragraph_answer']} <|im_end|> \n"
                f"<|im_start|>assistant {example['sentence_mask']} [SEP] {example['distract']}"
            )

        training_arguments = TrainingArguments(
            output_dir=self.checkpoint_path,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            num_train_epochs=self.num_epochs,
            evaluation_strategy="steps",
            eval_steps=0.2,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=1,
            logging_steps=1,
            warmup_steps=10,
            logging_strategy="steps",
            learning_rate=self.lr,
            fp16=False,
            bf16=False,
            group_by_length=True,
            report_to="wandb"
        )
        if self.type == 'qag':
            formatting_func = formatting_func_qag
        elif self.type == 'mcq':
            formatting_func = formatting_func_mcq
        else:
            formatting_func = formatting_func_gapfill

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = SFTTrainer(
            model=model,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            peft_config=peft_config,
            max_seq_length=self.max_len,
            tokenizer=tokenizer,
            packing=True,
            formatting_func=formatting_func,
            args=training_arguments,
        )

        trainer.train()

        wandb.finish()
        model.config.use_cache = True

    def compute_metrics(self):
        pass

    def generate(self, model_checkpoint: str, max_new_token: int = 512):
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        def save_result(path: str, result: dict[str, str]):
            file_mode = 'a' if os.path.exists(path) else 'w'
            with open(path, file_mode) as file:
                pd.DataFrame([result]).to_csv(file, index=False, header=(file_mode == 'w'))

        _, _, test_dataset = self.prepare_data()

        preds = []
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            prompt: str = (
                "<|im_start|> user \n"
                f"{sample['instruction']} "
                f"Context: {sample['paragraph']} <|im_end|> \n"
                f"<|im_start|>assistant"
            )

            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.cuda()
            outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_token, do_sample=True, top_p=0.75,
                                     temperature=0.1)

            pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
            #print(f"Prompt:\n{prompt}\n")
            print(f"Generated repsonse:\n{pred}")
            print(f"Ground truth:\n{sample['questions_answers']}\n\n")

            preds.append(pred)
            save_result(path=f'{self.checkpoint_path}/results.csv',
                        result={'prediction': pred, 'reference': sample['questions_answers']})

if __name__ == "__main__":
    fire.Fire(FineTuneLLMs)