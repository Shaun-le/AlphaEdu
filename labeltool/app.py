import streamlit as st
import pandas as pd
import requests

question_types = ['What', 'Who', 'When', 'Where', 'Why', 'How', 'Others']
Wh = [0, 0, 0, 0, 0, 0, 0]

def read_json(file_path):
    df = pd.read_json(file_path)
    return df

def update_file(df, file_path):
    df.to_json(file_path)

def find_first_sample_without_questions_answers(df):
    for index, row in df.iterrows():
        if not row['questions'] and not row['answers']:
            return index
    return None

st.set_page_config(layout="wide")

st.title('Label Tool for QG')
subject = ['Biology','Geography', 'GDCD', 'History', 'IT', 'Literature']
question_type_mapping = ['Sub', 'MCQs', 'Gap-fill']
file_path = ['data/visubqag_sub.json']

selected_type = st.selectbox('Type', question_type_mapping)

if selected_type == question_type_mapping[0]:
    sid = 0
    df = read_json(file_path[sid])
elif selected_type == question_type_mapping[1]:
    sid = 1
    df = read_json(file_path[sid])
else:
    sid = 2
    df = read_json(file_path[sid])

st.write(df)

if selected_type == question_type_mapping[0]:

    st.title('Subjective test')

    if len(df) == 0:
        st.write("No data available. Please add some paragraphs.")
    else:
        search_id = st.text_input("Enter ID to search:")

        if st.button('Search'):
            if search_id.isdigit():
                search_id = int(search_id)
                # Tìm kiếm theo cột ID
                sample_index = df[df['id'] == search_id].index
                if len(sample_index) > 0:
                    st.session_state['sample_index'] = sample_index[0]
                    st.experimental_rerun()
                else:
                    st.warning("ID not found!")
            else:
                st.warning("Please enter a valid ID!")

        sample_index = st.session_state.get('sample_index', 0)

        if sample_index >= len(df):
            st.write("No more samples available.")
        else:
            id = df['paragraphs'][sample_index]
            context = df['paragraphs'][sample_index]

            recon = st.text_area("Context", value=context)

            questions = st.session_state.get('questions', df.loc[sample_index, 'questions'])
            answers = st.session_state.get('answers', df.loc[sample_index, 'answers'])
            question_types_selected = st.session_state.get('question_types', [''] * len(questions))

            for i, (question, answer, question_type) in enumerate(zip(questions, answers, question_types_selected)):
                question_key = f"Question {i + 1}"
                answer_key = f"Answer {i + 1}"
                question_type_key = f"Question Type {i + 1}"

                if question_type in question_types:
                    question_type_index = question_types.index(question_type)
                else:
                    question_type_index = 0
                question_type_value = st.selectbox(question_type_key, question_types, index=question_type_index)

                col1, col2 = st.columns(2)
                with col1:
                    question_value = st.text_area(question_key, value=question)
                with col2:
                    answer_value = st.text_area(answer_key, value=answer)

                questions[i] = question_value
                answers[i] = answer_value
                question_types_selected[i] = question_type_value

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if st.button('Add'):
                    questions.append("")
                    answers.append("")
                    question_types_selected.append("")
                    st.session_state['questions'] = questions
                    st.session_state['answers'] = answers
                    st.session_state['question_types'] = question_types_selected
                    st.rerun()

            with col2:
                if st.button('Prev') and sample_index > 0:  # Ensure sample_index doesn't go below zero
                    sample_index -= 1
                    st.session_state['sample_index'] = sample_index
                    st.rerun()
            with col3:
                if st.button('Next') and sample_index < len(df) - 1:
                    sample_index += 1
                    st.session_state['sample_index'] = sample_index
                    st.rerun()
            with col5:
                if st.button('Delete'):
                    if len(df) > 0:
                        df.drop(index=sample_index, inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        if sample_index >= len(df):
                            sample_index = len(df) - 1
                        st.session_state['sample_index'] = sample_index
                        update_file(df, file_path[sid])
                    st.rerun()
            with col4:
                if st.button('Next Unannotated'):
                    next_unannotated_index = find_first_sample_without_questions_answers(df)
                    if next_unannotated_index is not None:
                        st.session_state['sample_index'] = next_unannotated_index
                    st.rerun()

            if st.button('Done'):
                for question_type in question_types_selected:
                    if question_type in question_types:
                        index = question_types.index(question_type)
                        Wh[index] += 1
                df.at[sample_index, 'paragraphs'] = recon
                df.at[sample_index, 'questions'] = questions
                df.at[sample_index, 'answers'] = answers
                df.at[sample_index, 'Wh'] = Wh
                update_file(df, file_path[sid])
                st.rerun()
elif selected_type == question_type_mapping[1]:
    st.title('Multiple Choice Questions')

else:
    st.title('Gap-fill')