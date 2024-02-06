from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import os
import sys
import time
import openai
import random
import logging

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from dotenv import dotenv_values
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

app = Flask(__name__)
api = Api(app)

# These three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override = True)
    config = dotenv_values(".env")

# Read environment variables
temperature = float(os.environ.get("TEMPERATURE", 0.7))
api_base = os.getenv("AZURE_OPENAI_BASE")
api_key = os.getenv("AZURE_OPENAI_KEY","3d21de1940a849b3bd4c97c710e35f2b")
api_type = os.environ.get("AZURE_OPENAI_TYPE", "azure")
api_version = os.environ.get("AZURE_OPENAI_VERSION", "2023-08-01-preview")
chat_completion_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
embeddings_deployment = os.getenv("AZURE_OPENAI_ADA_DEPLOYMENT")
model = os.getenv("AZURE_OPENAI_MODEL")
max_size_mb = int(os.getenv("CHAINLIT_MAX_SIZE_MB", 100))
max_files = int(os.getenv("CHAINLIT_MAX_FILES", 10))
text_splitter_chunk_size = int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE", 1000))
text_splitter_chunk_overlap = int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP", 0))
embeddings_chunk_size = int(os.getenv("EMBEDDINGS_CHUNK_SIZE", 16))
max_retries = int(os.getenv("MAX_RETRIES", 5))
backoff_in_seconds = float(os.getenv("BACKOFF_IN_SECONDS", 1))

# Configure system prompt
system_template = """Use the following pieces of context to answer the users question.\
"You are an intelligent assistant helping CGI Consultants who need quick and accurate access to internal documents" \
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.\
If you don't know the answer, just say that "I'm sorry, I don't know", don't try to make up an answer.Make sure you say "I don't know" in your response.\
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


# Configure OpenAI
openai.api_type = api_type
openai.api_version = api_version
openai.api_base = api_base
openai.api_key = api_key

# Set default Azure credential
default_credential = DefaultAzureCredential(
) if openai.api_type  ==  "azure_ad" else None

# Configure a logger
logging.basicConfig(stream = sys.stdout,
                    format = '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def backoff(attempt : int) -> float:
    return backoff_in_seconds * 2**attempt + random.uniform(0, 1)
        
# Create a Chroma vector store
embeddings = OpenAIEmbeddings(
    deployment="embedding",
    openai_api_key="3d21de1940a849b3bd4c97c710e35f2b",
    chunk_size = embeddings_chunk_size)

# Load Chroma vector store
db = Chroma(persist_directory="D365DocsDemo",embedding_function=embeddings)

# Create an AzureChatOpenAI llm
llm = AzureChatOpenAI(
    temperature = temperature,
    openai_api_key = "3d21de1940a849b3bd4c97c710e35f2b",
    openai_api_base = openai.api_base,
    openai_api_version = openai.api_version,
    openai_api_type = openai.api_type,
    deployment_name = chat_completion_deployment)

condensellm = AzureChatOpenAI(
    temperature=temperature,
    openai_api_key = "3d21de1940a849b3bd4c97c710e35f2b",
    openai_api_base = openai.api_base,
    openai_api_version = openai.api_version,
    openai_api_type = openai.api_type,
    deployment_name = chat_completion_deployment
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever= db.as_retriever(),         
#     condense_question_llm=condensellm,
#     memory = memory,
#     chain_type_kwargs = chain_type_kwargs
# )

chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs = chain_type_kwargs
    )

class Chatbot:
    def __init__(self):
        self.context = {}
    
    def chat(self, user_input):
        # Initialize the response
        response =  None

        # Retry the OpenAI API call if it fails
        for attempt in range(max_retries):
            try:
                # Ask the question to the chain
                response = chain({"question": user_input})
                break
            except openai.error.Timeout:
                # Implement exponential backoff
                wait_time = backoff(attempt)
                logger.exception(f"OpenAI API timeout occurred. Waiting {wait_time} seconds and trying again.")
                time.sleep(wait_time)
            except openai.error.APIError:
                # Implement exponential backoff
                wait_time = backoff(attempt)
                logger.exception(f"OpenAI API error occurred. Waiting {wait_time} seconds and trying again.")
                time.sleep(wait_time)
            except openai.error.APIConnectionError:
                # Implement exponential backoff
                wait_time = backoff(attempt)
                logger.exception(f"OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting {wait_time} seconds and trying again.")
                time.sleep(wait_time)
            except openai.error.InvalidRequestError:
                # Implement exponential backoff
                wait_time = backoff(attempt)
                logger.exception(f"OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting {wait_time} seconds and trying again.")
                time.sleep(wait_time)
            except openai.error.ServiceUnavailableError:
                # Implement exponential backoff
                wait_time = backoff(attempt)
                logger.exception(f"OpenAI API service unavailable. Waiting {wait_time} seconds and trying again.")
                time.sleep(wait_time)
            except Exception as e:
                logger.exception(f"A non retriable error occurred. {e}")
                break
        
        answer = response["answer"].strip("(")
        source_docs = response["source_documents"]
        sources = []
        for document in source_docs:
            if str(document.metadata["source"]) not in sources:
                sources.append(str(document.metadata["source"]))

        print(answer)
        print(response)
        if "I don't know" in answer:
            return answer
        if "I'm sorry" in answer:
            return answer
        #print(response("source_documents"))
        # Get the answer and sources from the response        
        return answer + "References:" + str(sources)

chatbot = Chatbot()

class ChatbotChat(Resource):
    def post(self):
        user_input = request.get_json().get('input', '') if request.get_json() else ''
        print(user_input)
        if not user_input:
            return jsonify({'error': 'No input provided'})
        result = chatbot.chat(user_input)
        return jsonify(result)

api.add_resource(ChatbotChat, '/chat')

if __name__ == '__main__':
    app.run(port=5002)
