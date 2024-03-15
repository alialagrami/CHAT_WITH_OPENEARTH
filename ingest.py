import os
from os import getenv
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yaml
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()
warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
    URLS = config["URLS"]
    print(URLS)


# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.
    """
    # Initialize loader

    url_loader = WebBaseLoader(URLS)
    url_loader.requests_per_second = 1
    loaded_documents = url_loader.aload()

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(loaded_documents)

    if not getenv("OPEN_AI_KEY"):
        # Initialize Ollama Embeddings
        ollama_embeddings = OllamaEmbeddings(model=os.environ["OLLAMA_MODEL"])

        # Create and persist a Chroma vector database from the chunked documents
        vector_database = Chroma.from_documents(
            documents=chunked_documents,
            embedding=ollama_embeddings,
            persist_directory=DB_DIR,
        )
    else:
        # Create OpenAI embeddings
        openai_embeddings = OpenAIEmbeddings()

        # Create a Chroma vector database from the documents
        vector_database = Chroma.from_documents(
            documents=chunked_documents,
            embedding=openai_embeddings,
            persist_directory=DB_DIR)
    vector_database.persist()


if __name__ == "__main__":
    create_vector_database()