import json
import numpy as np
import fire
from datasets import Dataset, load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Trainer, TrainingArguments, \
    Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torch
import os
import evaluate
from nltk.translate.bleu_score import sentence_bleu
import spacy
from loguru import logger
import nltk
nltk.download("wordnet")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlpvi = spacy.load('vi_core_news_lg')
#nlpen = spacy.load('en_core_web_sm')

def train(
        model: str = 'VietAI/vit5-base',
        data_path: str = 'shnl/ViLiteratureFake-MC',
        question_type: str = 'mcq', #'mcq', 'fill'
        max_target_len: int = 512,
        max_source_len: int = 1024,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        num_epochs: int = 10,
        num_proc: int = 8,
        checkpoint_path: str = './checkpoints/',
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Edu model with params:\n"
            f"model: {model}\n"
            f"data_path: {data_path}\n"
            f"question_type: {question_type}\n"
            f"max_target_len: {max_target_len}\n"
            f"max_source_len: {max_source_len}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"per_device_eval_batch_size: {per_device_eval_batch_size}\n"
            f"learning_rate: {learning_rate}\n"
            f"weight_decay: {weight_decay}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"num_epochs: {num_epochs}\n"
            f"num_proc: {num_proc}\n"
            f"checkpoint_path: {checkpoint_path}\n"
        )
    assert (
        model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    def formatting_mcp_func(example):
        input_seq = f"{example['answer']} [SEP] {example['question']} {example['paragraph']}"
        output_seq = f"{example['options']}"
        return {"input_seq" : input_seq, 'output_seq': output_seq}

    def formatting_fill_func(example):
        input_seq = f"{example['answer']} [SEP] {example['paragraph']}"
        output_seq = f"{example['sentence_answer_mask']} [SEP] {example['options']}"
        return {"input_seq" : input_seq, 'output_seq': output_seq}

    def bleu(predict, goal):
        bleu_scores = {1: [], 2: [], 3: [], 4: []}

        for sent1, sent2 in zip(predict, goal):
            sent1_doc = nlpvi(sent1)
            sent2_doc = nlpvi(sent2)
            ws = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
            for n in range(1, 5):
                weights = ws[n - 1]
                sent1_tokens = [token.text for token in sent1_doc]
                sent2_tokens = [token.text for token in sent2_doc]
                bleu_score = sentence_bleu([sent1_tokens], sent2_tokens, weights=weights)
                bleu_scores[n].append(bleu_score)
        result = {}
        for n in range(1, 5):
            avg_bleu_score = (sum(bleu_scores[n]) / len(bleu_scores[n])) * 100
            result["BLEU{}".format(n)] = (sum(bleu_scores[n]) / len(bleu_scores[n])) * 100
        return result

    def prepare_data():
        logger.info('-----:----- Preparing dataset -----:-----')
        data = load_dataset(f'{data_path}', use_auth_token=True)
        train_dataset = data['train']
        dev_dataset = data['validation']
        test_dataset = data['test']
        if question_type == 'mcq':
            train_dataset = train_dataset.map(formatting_mcp_func, num_proc=num_proc).remove_columns(
                ['paragraph', 'question', 'answer', 'options'])
            dev_dataset = dev_dataset.map(formatting_mcp_func, num_proc=num_proc).remove_columns(
                ['paragraph', 'question', 'answer', 'options'])
            test_dataset = test_dataset.map(formatting_mcp_func, num_proc=num_proc).remove_columns(
                ['paragraph', 'question', 'answer', 'options'])
        elif question_type == 'fill':
            train_dataset = train_dataset.map(formatting_fill_func, num_proc=num_proc).remove_columns(
                ['question', 'paragraph', 'answer', 'sentence', 'paragraph_sentence', 'paragraph_answer', 'sentence_answer', 'sentence_answer_mask', 'options'])
            dev_dataset = dev_dataset.map(formatting_fill_func, num_proc=num_proc).remove_columns(
                ['question', 'paragraph', 'answer', 'sentence', 'paragraph_sentence', 'paragraph_answer', 'sentence_answer', 'sentence_answer_mask', 'options'])
            test_dataset = test_dataset.map(formatting_fill_func, num_proc=num_proc).remove_columns(
                ['question', 'paragraph', 'answer', 'sentence', 'paragraph_sentence', 'paragraph_answer', 'sentence_answer', 'sentence_answer_mask', 'options'])
        return train_dataset, dev_dataset, test_dataset

    def compute_metric(
            model: AutoModelForSeq2SeqLM,
            tokenizer: AutoTokenizer,
            tokenized_test: Dataset
    ):
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
        dataloader = torch.utils.data.DataLoader(tokenized_test, collate_fn=data_collator,
                                                 batch_size=per_device_eval_batch_size)

        predictions = []
        references = []
        for _, batch in enumerate(tqdm(dataloader)):
            outputs = model.generate(
                input_ids=batch['input_ids'].to(device),
                max_length=max_target_len,
                attention_mask=batch['attention_mask'].to(device)
            )
            with tokenizer.as_target_tokenizer():
                outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out
                           in outputs]
                labels = np.where(batch['labels'] != -100, batch['labels'], tokenizer.pad_token_id)
                actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out
                           in labels]
                predictions.extend(outputs)
                references.extend(actuals)

        # results = metrics.compute(predictions=predictions, references=references)
        logger.info('-----:----- Dumping results -----:-----')
        with open(os.path.join(checkpoint_path, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump({'predictions': predictions, 'references': references}, f, ensure_ascii=False, indent=2)

        return {'predictions': predictions, 'references': references}

    def preprocess_and_train(
            model: AutoModelForSeq2SeqLM,
            tokenizer: AutoTokenizer,
            train_set: Dataset,
            val_set: Dataset,
            test_set: Dataset
    ):
        def preprocess_function(examples):
            inputs = tokenizer(examples["input_seq"], max_length=max_source_len, truncation=True, padding=True)
            labels = tokenizer(examples["output_seq"], max_length=max_target_len, truncation=True, padding=True)
            inputs['input_ids'] = inputs['input_ids']

            inputs['labels'] = labels['input_ids']

            return inputs

        logger.info('-----:----- Tokenizing datasets -----:-----')
        tokenized_train = train_set.map(preprocess_function, batched=True, remove_columns=['input_seq', 'output_seq'],
                                        num_proc=num_proc)
        tokenized_val = val_set.map(preprocess_function, batched=True, remove_columns=['input_seq', 'output_seq'],
                                    num_proc=num_proc)
        tokenized_test = test_set.map(preprocess_function, batched=True, remove_columns=['input_seq', 'output_seq'],
                                      num_proc=num_proc)

        with open(os.path.join(checkpoint_path, 'origin_refs.json'), 'w', encoding='utf-8') as f:
            json.dump({'references': test_set['output_seq']}, f, ensure_ascii=False, indent=2)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoint_path,
            do_train=True,
            do_eval=True,
            logging_strategy='steps',
            log_level='debug',
            logging_steps=500,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            predict_with_generate=True,
            group_by_length=True,
            eval_steps=500,
            evaluation_strategy='steps',
            save_strategy="steps",
            save_steps=50,
            save_total_limit=1,
            gradient_accumulation_steps=16,
            report_to='none',
            label_names=['labels']
        )
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
            eval_dataset=tokenized_val
        )
        logger.info('Training')
        trainer.train()

        outputs = compute_metric(model=model, tokenizer=tokenizer, tokenized_test=tokenized_test)

        return outputs

    os.makedirs(f'{checkpoint_path}', mode=0o777, exist_ok=True)
    logger.info(f'Load tokenizer and model checkpoint: {model}')
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)

    train_dataset, dev_dataset, test_dataset = prepare_data()

    outputs = preprocess_and_train(
        model=model,
        tokenizer=tokenizer,
        train_set=train_dataset,
        val_set=dev_dataset,
        test_set=test_dataset
    )

    with open(f'{checkpoint_path}/results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    pred = data['predictions']
    ref = data['references']

    bleu_score = bleu(pred, ref)

    metrics = load_metric('rouge')
    rouge_results = [{k: (v.mid.fmeasure) * 100} for k, v in metrics.compute(predictions=pred, references=ref).items()]

    meteor_metrics = evaluate.load('meteor')
    meteor_results = meteor_metrics.compute(predictions=pred, references=ref)

    bert_score = evaluate.load('bertscore')
    bert_results = bert_score.compute(predictions=pred, references=ref, lang='vi')
    bert_mean_f1 = np.array(bert_results['f1']).mean()

    results = {
        "bleu_score": bleu_score,
        "rouge_results": rouge_results,
        "meteor_results": meteor_results,
        "bert_score_mean_f1": bert_mean_f1
    }

    print(results)

    with open(f'{checkpoint_path}/scores.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    fire.Fire(train)