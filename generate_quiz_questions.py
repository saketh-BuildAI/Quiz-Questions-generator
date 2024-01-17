import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, create_extraction_chain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
import pickle
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import openai
import json
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)
memory = ConversationBufferMemory()


def sort_objects(obj_list):
    question = []
    options = []
    correct = []

    for obj in obj_list:

        if 'question' in obj:
            question.append(obj['question'])
        for i in range(3):
            list = []
            if 'option1' in obj:
                list.append(obj['option1'])
            if 'option2' in obj:
                list.append(obj['option2'])
            if 'option3' in obj:
                list.append(obj['option3'])
        options.append(list)
        if 'correct answer' in obj:
            correct.append(obj['correct answer'])

    return [question, options, correct]


def create_ques_ans(number_of_qn, topic, standard):
    if standard == "Basic":
        level = "Remembering, Understanding"
    if standard == "Intermediate":
        level = "Applying, Analyzing"
    if standard == "Advanced":
        level = "Evaluating or complex numerical"

    template = template = f"""Create {number_of_qn} multiple choice questions on {topic} with 3 options
    in {level} levels of blooms taxonomy. """

    llm = ChatOpenAI(model='gpt-3.5-turbo')

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    with open(os.getcwd() + '/Vector_DB/CBSE-9th-Motion.pkl', 'rb') as f:
        chunks = pickle.load(f)
    db = Chroma.from_texts(chunks, embedding=embeddings)
    qa_chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    matching_docs = db.similarity_search(template)
    answer = qa_chain.run(input_documents=matching_docs, question=template)

    schema = {
        "properties": {
            "question": {"type": "string"},
            "option1": {"type": "string"},
            "option2": {"type": "string"},
            "option3": {"type": "string"},
            "correct answer": {"type": "string"}
        },
        "required": ["question", "options", "correct_answer"]
    }

    llm2 = ChatOpenAI(model="gpt-3.5-turbo-0613")
    chain = create_extraction_chain(schema, llm2)
    response = chain.run(answer)

    return sort_objects(response)


def remove_control_characters(content):
    # Define a translation table to remove specific control characters
    control_characters = bytes([0x98, 0x99, 0x80, 0x93])
    translation_table = dict.fromkeys(control_characters, None)

    # Use translate to remove control characters
    cleaned_content = content.translate(translation_table)
    return cleaned_content


number_of_qn = 3
topic = "Quantum Mechanics"
standard = "Basic"
data = create_ques_ans(number_of_qn, topic, standard)
questions = data[0]
options = data[1]
answers = data[2]

questions_data = []
for i in range(number_of_qn):
    info = {
        "Question": remove_control_characters(str(questions[i])),
        "Options": remove_control_characters(str(options[i])),
        "Answer": remove_control_characters(str(answers[i]))
    }
    questions_data.append(info)

json_data = json.dumps(questions_data, indent=2)
print(json_data)
file_path = 'questions_data.json'
with open(file_path, 'w') as file:
    json.dump(questions_data, file, indent=2)
