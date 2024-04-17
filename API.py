from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

model_id = {
    'qg': 'namngo/pipeline-vit5-qg',
    'ag': 'namngo/pipeline-vit5-viquad-ae',
    'qag': 'shnl/vit5-vinewsqa-qg-ae',
}

def preprocess_function(examples, tokenizer):
    inputs = tokenizer(
        examples["input_seq"],
        max_length=1024,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    return {key: tensor.to('cuda') for key, tensor in inputs.items()}

def format_qa(list_raw_string):
    qa_pairs = set()
    for raw_string in list_raw_string:
        if ", answer: " not in raw_string or "question: " not in raw_string:
            print("Error: Invalid QA format")
        else:
            q, a = map(str.strip, raw_string.split(", answer: "))
            qa_pairs.add((q, a))
    return list(qa_pairs)

def generate_q(name, context, answer):
    tokenizer = AutoTokenizer.from_pretrained(model_id[name])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id[name])
    instructions_qg = [
        'Hãy tạo ra câu hỏi từ nội dung của ngữ cảnh và câu trả lời dưới đây.',
        'Sử dụng thông tin sau để đặt câu hỏi có thể dẫn đến câu trả lời tương ứng.',
        'Hãy tạo một câu hỏi dựa trên thông tin sau.',
        'Đặt ra câu hỏi phù hợp với ngữ cảnh và câu trả lời dưới đây.',
        'Từ ngữ cảnh sau đây, hãy đặt câu hỏi để có câu trả lời như bên dưới.',
        'Từ ngữ cảnh và câu trả lời sau, hãy sinh ra câu hỏi tương ứng.',
        'Đặt câu hỏi phù hợp cho thông tin sau đây.',
        'Nên đặt câu hỏi thế nào cho ngữ cảnh để nhận được câu trả lời sau.',
        'Sinh ra câu hỏi để đáp ứng được câu trả lời và ngữ cảnh sau đây.',
        'Từ các nội dung được cung cấp, sinh ra câu hỏi tương ứng.',
        'Dựa vào nội dung và câu trả lời sau, hãy tạo câu hỏi phù hợp.',
        'Sinh ra câu hỏi đáp ứng được câu trả lời và ngữ cảnh sau đây.',
        'Đặt câu hỏi phù hợp với câu trả lời cùng với ngữ cảnh sau đây.'
    ]
    instruction_qg = random.choice(instructions_qg)
    examples = {"input_seq": str("### Instruction: \n"
                                 f"{instruction_qg} \n\n"
                                 f"### Context: \n"
                                 f"{context}\n\n"
                                 f"### Answer: \n"
                                 f"{answer} \n\n"
                                 f"\n\n### Response: \n")}
    new_inputs = preprocess_function(examples, tokenizer)
    max_target_length = 256
    model = model.to('cuda')
    with torch.no_grad():
        new_outputs = model.generate(
            input_ids=new_inputs['input_ids'].to('cuda'),
            max_length=max_target_length,
            attention_mask=new_inputs['attention_mask'].to('cuda'),
        )
    with tokenizer.as_target_tokenizer():
        output = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out
                  in new_outputs]
    return output

def generate_a(name, context, question):
    tokenizer = AutoTokenizer.from_pretrained(model_id[name])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id[name])
    instructions_ag = [
        'Hãy tạo ra câu trả lời đúng với nội dung của ngữ cảnh và question dưới đây.',
        'Sử dụng thông tin sau để trả lời câu hỏi tương ứng.',
        'Hãy tạo câu trả lời dựa trên thông tin sau.',
        'Từ ngữ cảnh và câu hỏi sau, hãy sinh ra câu trả lời đúng.',
        'Trả lời câu hỏi sau sao cho phù hợp với nội dung của ngữ cảnh.',
        'Trả lời câu hỏi bên dưới từ nội dung của ngữ cảnh.',
        'Từ nội dung của ngữ cảnh cùng với question, hãy sinh ra câu trả lời phù hợp.',
        'Hãy sinh ra câu trả lời cho câu hỏi và ngữ cảnh bên dưới.',
        'Sử dụng thông tin từ ngữ cảnh và câu trả lời bên dưới để sinh ra câu hỏi phù hợp.',
        'Sinh ra câu trả lời đúng với câu hỏi và phù hợp với ngữ cảnh sau đây.',
        'Sinh ra câu trả lời từ nội dung của ngữ cảnh sao cho phù hợp với câu hỏi sau đây.',
        'Trả lời câu hỏi từ nội dung ngữ cảnh sau đây.',
        'Sinh ra câu trả lời đúng và phù hợp với nội dung ngữ cảnh và câu hỏi sau đây.'
    ]
    instruction_ag = random.choice(instructions_ag)
    examples = {"input_seq": str("### Instruction: \n"
                                 f"{instruction_ag} \n\n"
                                 f"### Context: \n"
                                 f"{context}\n\n"
                                 f"### Question: \n"
                                 f"{question} \n\n"
                                 f"\n\n### Response: \n")}
    new_inputs = preprocess_function(examples, tokenizer)
    max_target_length = 256
    model = model.to('cuda')
    with torch.no_grad():
        new_outputs = model.generate(
            input_ids=new_inputs['input_ids'].to('cuda'),
            max_length=max_target_length,
            attention_mask=new_inputs['attention_mask'].to('cuda'),
        )
    with tokenizer.as_target_tokenizer():
        output = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out
                  in new_outputs]
    return output

def generate_qa(name, context):
    unique_qa_pairs = set()
    tokenizer = AutoTokenizer.from_pretrained(model_id[name])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id[name])
    instructions = ['Xây dựng một số cặp câu hỏi và câu trả lời dựa trên đoạn văn bản sau.',
                'Tạo ra một vài cặp câu hỏi và câu trả lời tương ứng với đoạn văn bản sau.',
                'Liệt kê một số câu hỏi và câu trả lời tương ứng với đoạn văn bản sau.',
                'Cho biết một số câu hỏi và câu trả lời tương ứng với đoạn văn bản sau.',
                'Viết ra một số câu hỏi và câu trả lời tương ứng với đoạn văn bản sau.',
                'Hãy đặt một số câu hỏi và câu trả lời cho đoạn văn sau.',
                'Hãy xây dựng cặp câu hỏi và câu trả lời liên quan đến đoạn văn sau.',
                'Hãy tạo ra các cặp câu hỏi và câu trả lời dựa trên nội dung của đoạn văn sau.',
                'Hãy phát triển một số cặp câu hỏi và câu trả lời liên quan đến thông tin trong đoạn văn sau.',
                'Tạo các cặp câu hỏi và câu trả lời liên quan đến đoạn văn bản sau.',
                'Viết các cặp câu hỏi và câu trả lời dựa trên đoạn văn bản sau.',
                'Đặt ra các câu hỏi và câu trả lời phù hợp với đoạn văn bản sau.',
                'Thiết kế các câu hỏi và câu trả lời tương ứng với nội dung đoạn văn bản sau.']
    instruction_qa = random.choice(instructions)
    examples = {"input_seq": str("### Instruction: \n"
                                 f"{instruction_qa} \n\n"
                                 f"### Input: \n"
                                 f"{context}\n\n"
                                 f"\n\n### Response: \n")}
    new_inputs = preprocess_function(examples, tokenizer)
    max_target_length = 256
    model = model.to('cuda')
    with torch.no_grad():
        new_outputs = model.generate(
            input_ids=new_inputs['input_ids'].to('cuda'),
            max_length=max_target_length,
            attention_mask=new_inputs['attention_mask'].to('cuda'),
        )

    with tokenizer.as_target_tokenizer():
        new_outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out
                       in new_outputs]
        out = [format_qa(o.split('[SEP]')) for o in new_outputs]
    pred = ''
    for item in out[0]:
        question, answer = item
        if (question, answer) not in unique_qa_pairs:
            pred += f'{question} \nanswer: {answer} [*] '
            unique_qa_pairs.add((question, answer))
        pred = pred.replace('_', ' ').replace(' ?', '?')
    qag = '\n\n'.join(pred.split(' [*] '))
    return qag

@app.route('/gen', methods=['POST'])
def receive_data():
    context = request.json.get('context')
    name = request.json.get('name')
    if name == 'qg':
        answer = request.json.get('answer')
        print(answer)
        output = generate_q(name, context, answer)
    elif name == 'ag':
        question = request.json.get('question')
        output = generate_a(name, context, question)
    else:
        output = generate_qa(name, context)

    return jsonify({'output': output})


if __name__ == '__main__':
    app.run(debug=True, port=7777)