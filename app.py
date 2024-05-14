from flask import Flask, request, jsonify
from flask_cors import CORS
from plms.language_model import TransformersQG
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Trainer, TrainingArguments, \
    Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch
import re
from tqdm import tqdm
from datasets import Dataset
import random
import requests

app = Flask(__name__)
CORS(app)

model_map = {
    'Subjective Test': 'shnl/vit5-vinewsqa-qg-ae',
    'Multiple Choice Question': 'shnl/BARTPho-ViBiologyFake-MCQG',
    'Fill in the Blank': 'fill'
}

def generateQAG(model, context):
    unique_qa_pairs = set()
    model = TransformersQG(model=model, max_length=512)
    out = model.generate_qa(context)
    pred = ''
    for item in out:
        question, answer = item
        if (question, answer) not in unique_qa_pairs:
            pred += f'question: {question} \nanswer: {answer} [*] '
            unique_qa_pairs.add((question, answer))
    qag = '\n\n'.join(pred.split(' [*] '))
    return qag

def generateA(context):
    model = TransformersQG(model='namngo/pipeline-vit5-viquad-ae')
    out = model.generate_a(context)
    return out

def generateQ(context, list_answer):
    list_context = [context] * len(list_answer)
    model = TransformersQG(model='shnl/vit5-vinewsqa-qg-ae')
    out = model.generate_q(list_context, list_answer)
    return out

def format_mcq_string(mcq_list):
    formatted_string = ""
    for q, mcq in mcq_list:
        formatted_string += f"{q}\n{mcq}\n\n"
    return formatted_string
def formatting_mcq(list_question, list_answer, list_distraction):
    if len(list_question) != len(list_answer) != len(list_distraction):
        raise ValueError("Something Wrong!")

    mcq_list = []
    for q, a, d in zip(list_question, list_answer, list_distraction):
        ds = d.split('[SEP]')
        if len(ds) > 3:
            ds = ds[:3]
        ds.append(a)
        random.shuffle(ds)
        mcq_options = ['A', 'B', 'C', 'D']
        mcq = '\n'.join([f"{mcq_options[i]}. {option}" for i, option in enumerate(ds)])
        mcq_list.append((q, mcq))
    return mcq_list

def find_and_replace(sentence, answer):
    masked_sentence = re.sub(r'\b' + re.escape(answer) + r'\b', '____', sentence)
    sentences = masked_sentence.split('.')
    sentences_with_mask = [s.strip() for s in sentences if '____' in s]
    return sentences_with_mask


def formatting_fill(context, list_answer, list_distraction):
    if len(list_answer) != len(list_distraction):
        raise ValueError("The lengths of answer and distraction lists must match.")

    mcq_list = []
    for a, d in zip(list_answer, list_distraction):
        ds = d.split('[SEP]')
        if len(ds) > 3:
            ds = ds[:3]
        ds.append(a)
        random.shuffle(ds)
        mcq_options = ['A', 'B', 'C', 'D']
        mcq = '\n'.join([f"{mcq_options[i]}. {option}" for i, option in enumerate(ds)])
        am = find_and_replace(context, a)
        if len(am) > 1:
            for sentence in am:
                mcq_list.append((sentence, mcq))
        else:
            mcq_list.append((am[0], mcq))

    return mcq_list

def formatting_mcq_func(example):
    input_seq = f"{example['answer']} [SEP] {example['question']} {example['paragraph']}"
    return {"input_seq": input_seq}

def generateMC1(list_answer, list_question, context, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    list_context = [context] * len(list_answer)
    outputs = []
    dict_obj = {'paragraph': list_context, 'answer': list_answer, 'question': list_question}
    datasets = Dataset.from_dict(dict_obj)
    new_inputs = datasets.map(formatting_mcq_func, num_proc=1).remove_columns(['question', 'paragraph', 'answer'])
    def preprocess_function(examples):
        inputs = tokenizer(examples["input_seq"], max_length=1024, truncation=True, padding=True)
        inputs['input_ids'] = inputs['input_ids']
        return inputs
    tokenized_test = new_inputs.map(preprocess_function, batched=True, remove_columns=['input_seq'],num_proc=1)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    dataloader = torch.utils.data.DataLoader(tokenized_test, collate_fn=data_collator,
                                             batch_size=2)
    predictions = []
    for _, batch in enumerate(tqdm(dataloader)):
        outputs = model.generate(
            input_ids=batch['input_ids'],
            max_length=512,
            attention_mask=batch['attention_mask']
        )
        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out
                       in outputs]
            predictions.append(outputs)
    return predictions

def generateMC(context,list_question,list_answer):
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'
    headers = {'Content-Type': 'application/json'}
    outputs = []
    for qus, anw in zip(list_question,list_answer):
        data = {
            'contents': [
                {
                    'parts': [
                        {
                            'text': f"Hãy sinh ra 3 đáp án sai (distraction) từ nội dung sau (Các đáp án cách nhau bằng [SEP] ): Context: {context}, Question: {qus} Answer: {anw}"
                        }
                    ]
                }
            ]
        }
        api_key = 'AIzaSyApFAbCUA1H-VHAidzqmyStHFe92ODeO1Y'
        params = {'key': api_key}
        response = requests.post(url, headers=headers, json=data, params=params)
        correct = response.json()['candidates'][0]['content']['parts'][0]['text']
        outputs.append(correct)
    return outputs

def generate_output(context, models):
    output = ""
    for model in models:
        if model == 'shnl/vit5-vinewsqa-qg-ae':
            output = generateQAG(model, context)
        elif model == 'shnl/BARTPho-ViBiologyFake-MCQG':
            list_answer = generateA(context)
            list_question = generateQ(context, list_answer)
            list_distraction = generateMC(context, list_question, list_answer)
            output = format_mcq_string(formatting_mcq(list_question, list_answer, list_distraction))
        elif model == 'fill':
            list_answer = generateA(context)
            list_question = generateQ(context, list_answer)
            list_distraction = generateMC(context, list_question, list_answer)
            output = format_mcq_string(formatting_fill(context, list_answer, list_distraction))
    return output
@app.route('/gen', methods=['POST'])
def generate_questions():
    data = request.json
    context = data.get('context')
    selected_items = data.get('selectedItems')
    models = [model_map[item] for item in selected_items]
    output = generate_output(context, models)
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True, port=8888)