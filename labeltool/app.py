import streamlit as st
import pandas as pd
import requests

st.set_page_config(layout="wide")

st.title('Label Tool for Alpha Edu')

def generateMC(context, question, answer):
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'
    headers = {'Content-Type': 'application/json'}
    data = {
        'contents': [
            {
                'parts': [
                    {
                        'text': f"You are a great biologist, here is the following content: context: '{context}', question: '{question}', answer: '{answer}' generate three distract answers. Distractor answers are separated by [SEP]. Example: Distract answer 1 [SEP] Distract answer 2 [SEP] Distract answer 3"
                    }
                ]
            }
        ]
    }
    api_key = 'AIzaSyApFAbCUA1H-VHAidzqmyStHFe92ODeO1Y'
    params = {'key': api_key}
    response = requests.post(url, headers=headers, json=data, params=params)
    if response.status_code == 200:
        correct = response.json()['candidates'][0]['content']['parts'][0]['text']
        return correct
    else:
        return "Failed to generate distractors"

def read_data(file_path):
    data = pd.read_json(file_path, encoding='utf-8')
    return data

def update_data(file_path, data):
    data.to_json(file_path, force_ascii=False)

def find_next_unannotated_index(data):
    for i, row in data.iterrows():
        if all(row[q_type] == 0 for q_type in question_types):
            return i
    return None

file_path_map = {
    'Biology': 'data/BiologyQA.json',
    'Geography': 'data/GeographyQA.json'
}

question_types = ['What', 'Who', 'When', 'Where', 'Why', 'How', 'Others']
subject = st.selectbox('Choose a subject:', options=list(file_path_map.keys()))
file_path = file_path_map[subject]
data = read_data(file_path)
st.dataframe(data)

if 'sample_index' not in st.session_state:
    st.session_state['sample_index'] = find_next_unannotated_index(data)

if 'sample_index' in st.session_state:
    row_index = st.session_state['sample_index']
    selected_row = data.iloc[row_index]
    st.write(f"Sample: {row_index + 1} / {len(data)}")
    context = st.text_area("Context:", value=selected_row['context'], height=150)
    col1, col2, col3 = st.columns(3)
    with col1:
        question = st.text_area("Question:", value=selected_row['question'], height=20)
        data.at[row_index, 'question'] = question
    with col2:
        answer = st.text_area("Answer:", value=selected_row['answer'], height=20)
        data.at[row_index, 'answer'] = answer
    with col3:
        distract = st.text_area("Distract:", value=selected_row['distract'], height=20)
        data.at[row_index, 'distract'] = distract

    columns = st.columns(len(question_types))
    for i, q_type in enumerate(question_types):
        if q_type not in data.columns:
            data[q_type] = ''
        new_count = columns[i].text_area(q_type + ':', value=selected_row.get(q_type, ''), key=f"{q_type}_{row_index}", height=10)
        if new_count != selected_row.get(q_type, ''):
            data.at[row_index, q_type] = new_count
            update_data(file_path, data)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col3:
    if st.button('Next'):
        if 'sample_index' in st.session_state:
            st.session_state['sample_index'] = (st.session_state['sample_index'] + 1) % len(data)
        st.rerun()
with col2:
    if st.button('Next Unannotated'):
        next_unannotated_index = find_next_unannotated_index(data)
        if next_unannotated_index is not None:
            st.session_state['sample_index'] = next_unannotated_index
        st.rerun()
with col1:
    if st.button('Prev'):
        if 'sample_index' in st.session_state:
            st.session_state['sample_index'] = (st.session_state['sample_index'] - 1) % len(data)
        st.rerun()
with col4:
    if st.button('Generate Distraction'):
        new_distract = generateMC(context, question, answer)
        if new_distract != "Failed to generate distractors":
            data.at[row_index, 'distract'] = new_distract
            update_data(file_path, data)
        else:
            st.error("Failed to generate distractors. Please check API and inputs.")
        st.rerun()
with col5:
    if st.button('Delete'):
        data = data.drop(index=row_index)
        data.reset_index(drop=True, inplace=True)
        update_data(file_path, data)
        st.rerun()

#with col6:
    #sample_index_input = st.text_input("Go to Sample Index:")
    #if sample_index_input:
        #sample_index_input = int(sample_index_input)
        #if 0 <= sample_index_input < len(data):
           #st.session_state['sample_index'] = sample_index_input

#with col7:
    #if st.button('RERUN'):
        #st.rerun()