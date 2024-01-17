from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
import json
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


class Option(BaseModel):
    text: str = Field(description="The text of the option.")
    correct: str = Field(description="Whether the option is correct or not. Either 'true' or 'false'")


class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question")
    options: List[Option] = Field(description="The possible answers to the question.")


class Quiz(BaseModel):
    MCQ: List[QuizQuestion] = Field(description="List of questions with answers")


parser = PydanticOutputParser(pydantic_object=Quiz)

llm = ChatOpenAI()
format_instructions = parser.get_format_instructions()
num = 3
result = llm.predict(f"Give me {num} quiz questions on ancient history and four options. \n {format_instructions}")
structured_output = parser.parse(result)

print(structured_output.MCQ[0].options)
questions = []
for i in range(num):
    info = {
        "Question": str(structured_output.MCQ[i].question),
        "Options": str(structured_output.MCQ[i].options),

    }
    questions.append(info)

json_data = json.dumps(questions, indent=2)
print(json_data)
file_path = 'questions.json'
with open(file_path, 'w') as file:
    json.dump(questions, file, indent=2)
