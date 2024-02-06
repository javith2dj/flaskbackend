# Import packages
import os
import io
import sys
import time
import openai
import random
import logging
from pypdf import PdfReader
from docx import Document
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
api_key = os.getenv("AZURE_OPENAI_KEY")
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

chain = None

# Configure system prompt
system_template = """Use the following pieces of context to answer the users question.\
"You are an intelligent assistant helping CGI Consultants who need quick and accurate access to internal documents" \
"Answer the question using only the data provided in the information sources below. " \
"Each source has a name followed by colon and the actual data, quote the source name for each piece of data you use in the response. " \
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
"For example, if the question is \"What color is the sky?\" and one of the information sources says \"info123: the sky is blue whenever it's not cloudy\", then answer with \"The sky is blue [info123]\" " \
"It's important to strictly follow the format where the name of the source is in square brackets at the end of the sentence, and only up to the prefix before the colon (\":\"). " \
"If there are multiple sources, cite each one in their own square brackets. For example, use \"[info343][ref-76]\" and not \"[info343,ref-76]\". " \
"Never quote tool names as sources." \
"If you cannot answer using the sources below, say that you don't know. " \
"\n\nYou can access to the following tools:"

Example of your response should be:

---

The answer is foo
SOURCES: xyz

---

Begin!
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

def llmstart():
    global chain
    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings(
        deployment="embedding",
        openai_api_key="ffd6ab9f00c94c52b665b53a25f5df5f",
        chunk_size = embeddings_chunk_size)

    # Load Chroma vector store
    db = Chroma(persist_directory="FDDDocs_Rec",embedding_function=embeddings)

    # Create an AzureChatOpenAI llm
    llm = AzureChatOpenAI(
        temperature = temperature,
        openai_api_key = openai.api_key,
        openai_api_base = openai.api_base,
        openai_api_version = openai.api_version,
        openai_api_type = openai.api_type,
        deployment_name = chat_completion_deployment)
    
    condensellm = AzureChatOpenAI(
        temperature=temperature,
        openai_api_key = openai.api_key,
        openai_api_base = openai.api_base,
        openai_api_version = openai.api_version,
        openai_api_type = openai.api_type,
        deployment_name = chat_completion_deployment
    )

    #Create a chain that uses the Chroma vector store
    # chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm = llm,
    #     chain_type = "stuff",
    #     retriever = db.as_retriever(),
    #     return_source_documents = True,
    #     chain_type_kwargs = chain_type_kwargs
    # )



    #compressor = LLMChainExtractor.from_llm(llm)
    #compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=db.as_retriever())
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history",return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= db.as_retriever(),         
        condense_question_llm=condensellm,
        memory = memory
    )

    #chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=db.as_retriever,memory=memory);

def ask(message: str)->str:
    # Initialize the response
    response =  None

    # Retry the OpenAI API call if it fails
    for attempt in range(max_retries):
        try:
            # Ask the question to the chain
            response = chain(message)
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

    # Get the answer and sources from the response
    answer = response["answer"]
    return answer