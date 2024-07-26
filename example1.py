import os
from constants import openapi_key
from langchain.llms import openai
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st 

os.environ["OPENAI_API_KEY"] = openapi_key

st.title('Celebrity Search Results')
input_text=st.text_input("Enter a celebrity name")

first_input_prompt = PromptTemplate(
    input_variable=['name'],
    template="Tell me more about {name}.",
)

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

