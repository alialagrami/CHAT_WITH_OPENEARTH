import os
from os import environ
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
import re
CONTEXT_PATTERN = re.compile(r"^CONTEXT:")


def _run(self, tool_input: str) -> str:
    """Use the tool."""
    CONTEXT_PROMPT = "CONTEXT:You must ask the human about {context}. Reply with schema #2."
    if isinstance(tool_input, str):
        product_tag, firmware_version, query = tool_input.split(",")

    if product_tag == "0":
        return CONTEXT_PROMPT.format(context="the product name")
    if firmware_version == "0":
        return CONTEXT_PROMPT.format(context="the firmware version")


system_template = """
Use the following pieces of context to answer the users question and strictly follow it. 
you can ONLY help in gardening and plants related topics as a plant garden advisor,If you don't know the answer, just say that you don't know, don't try to make up an answer.
YOU SHOULD keep asking clarification QUESTIONS to the user UNTIL YOU HAVE ALL THE INFORMATION REQUIRED to 
answer the question asked by the user.
"""

# Load environment variables from .env file (Optional)
load_dotenv()
USE_OPENAI = False
if environ.get("USE_OPENAI"):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL = os.getenv("LANGCHAIN_PROJECT")
    USE_OPENAI = True
else:
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
    MODEL = os.getenv("OLLAMA_MODEL")

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def main():
    # Set the title and subtitle of the app
    st.title('Help your Plant OPEN EARTH')
    prompt = st.text_input("Ask a question (query/prompt)")
    ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
    DB_DIR: str = os.path.join(ABS_PATH, "db")

    if not USE_OPENAI:
        # Create Ollama embeddings
        ollama_embeddings = OllamaEmbeddings(model=MODEL)
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=ollama_embeddings)
        llm = Ollama(model=MODEL)
    else:
        # Create OpenAI embeddings
        openai_embeddings = OpenAIEmbeddings()
        # Create a Chroma vector database from the documents
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=openai_embeddings)
        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name=MODEL)
    # Create a retriever from the Chroma vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    # Create a RetrievalQA from the model and retriever
    prompt2 = PromptTemplate.from_template(system_template)
    question_generator_chain = LLMChain(llm=llm, prompt=prompt2)
    agent_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory
    )

    if prompt:
        try:
            response = agent_chain.run(prompt)
        except Exception as error:
            print(error)
        finally:
            print(response)
            st.write(response)


if __name__ == '__main__':
    main()