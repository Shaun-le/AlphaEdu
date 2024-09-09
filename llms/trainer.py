from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb, fire, evaluate
from loguru import logger
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

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
    def __init__(self):
        self.pretrained_model_name_or_path: str = 'SeaLLMs/SeaLLMs-v3-1.5B',
        self.checkpoint_path: str = './cp',
        self.max_len: int = 2048,
        self.seed: int = 42,
        self.num_proc: int = 16,
        self.data_path: str = 'shnl/ViHistory',
        self.per_device_train_batch_size: int = 4,
        self.per_device_eval_batch_size: int = 8,
        self.num_epochs: int = 10,
        self.lr: float = 1e-4,
        self.warmup_ratio: float = 0.05,
        self.weight_decay: float = 0.015,
        self.use_flash_attention: bool = False,
        self.type: str = 'qag',  # mcq gap-fill
    def prepare_data(self, tokenizer):
        logger.info('-----:----- Preparing dataset -----:-----')
        dataset = load_dataset(f'{self.data_path}', use_auth_token=True)
        train_dataset = dataset['train']
        dev_dataset = dataset['validation']
        test_dataset = dataset['test']

        def formatting_func_qag(example):
            row_json = [{"role": "system", "content": "You are a "},
                        {"role": "user", "content": example['paragraph']},
                        {"role": "assistant", "content": example['question&answer']}]
            example["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
            return example

        def formatting_func_mcq(example):
            row_json = [{"role": "system", "content": "You are a "},
                        {"role": "user", "content": example['paragraph'] + example['answer'] + example['question']},
                        {"role": "assistant", "content": example['distractor']}]
            example["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
            return example

        def formatting_func_gf(example):
            row_json = [{"role": "system", "content": "You are a "},
                        {"role": "user", "content": example['paragraph_answer']},
                        {"role": "assistant", "content": f"{example['sentence_mask']} [SEP] {example['distractor']}"}]
            example["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
            return example

        if type == 'qag':
            train_dataset = train_dataset.map(formatting_func_qag, num_proc=self.num_proc)
            dev_dataset = dev_dataset.map(formatting_func_qag, num_proc=self.num_proc)
            test_dataset = test_dataset.map(formatting_func_qag, num_proc=self.num_proc)
        elif type == 'mcq':
            train_dataset = train_dataset.map(formatting_func_mcq, num_proc=self.num_proc)
            dev_dataset = dev_dataset.map(formatting_func_mcq, num_proc=self.num_proc)
            test_dataset = test_dataset.map(formatting_func_mcq, num_proc=self.num_proc)
        else:
            train_dataset = train_dataset.map(formatting_func_gf, num_proc=self.num_proc)
            dev_dataset = dev_dataset.map(formatting_func_gf, num_proc=self.num_proc)
            test_dataset = test_dataset.map(formatting_func_gf, num_proc=self.num_proc)

        return train_dataset, dev_dataset, test_dataset
    def train(
            self,
    ):
        print(
            f"Training Edu llms model with params:\n"
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

        torch_dtype = torch.float16
        attn_implementation = "eager"

        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        model, tokenizer = setup_chat_format(model, tokenizer)

        # LoRA config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )
        model = get_peft_model(model, peft_config)

        train_dataset, dev_dataset, test_dataset = self.prepare_data(tokenizer)

        training_arguments = TrainingArguments(
            output_dir=self.checkpoint_path,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            num_train_epochs=1,
            evaluation_strategy="steps",
            eval_steps=0.2,
            logging_steps=1,
            warmup_steps=10,
            logging_strategy="steps",
            learning_rate=2e-4,
            fp16=False,
            bf16=False,
            group_by_length=True,
            report_to="wandb"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            peft_config=peft_config,
            max_seq_length=512,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False,
        )

        trainer.train()

    def compute_metrics(self):
        pass
    def generate(self):
        pass


if __name__ == "__main__":
    fire.Fire(FineTuneLLMs)