import os
import openai
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from datasets import load_dataset
import csv
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

#----------------------------------------------------------------------------

#load open source embedding 
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs = {'device': 'cpu'})
# load FAISS vectorstore, which was saved in python notebook for test evaluation
vectorstore=FAISS.load_local("vectorstore", embeddings ,allow_dangerous_deserialization=True)

#load claude-3 model
#os.environ["ANTHROPIC_API_KEY"] = "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"
#llm = ChatAnthropic(model="claude-3-sonnet-20240229")

#load mistral LLM
os.environ["MISTRAL_API_KEY"] = "d1GrjMjjhRwbhu3KijtoM5P75oF07WH3"
llm = ChatMistralAI(model="mistral-large-latest")

#keeping k as 5, to limit last 5 chat history only
def filter_messages(messages, k=5):
    return messages[-k:]

#stores all the chat message history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


#-------------------------RAG----------------------------------------------------------------

#prompt template for rag based generation
prompt_rag = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are expert in writing human like answers for an asked question.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


#make a chain
chain_rag = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt_rag
    | llm
)

#final function of generation which includes generation chain and message history
with_message_history_rag = RunnableWithMessageHistory(chain_rag, get_session_history, input_messages_key="messages")

def generation_qa_pair(context_documents: list):
    """
    This function takes in a list of retrieved docs and return string of question answer pair for the highest similar retrieved question
    """
    question_answer_context=f"""Question: \n {context_documents[0].page_content} \n Answer Examples:"""
    for index, answer in enumerate(context_documents[0].metadata['top_answers']):
        question_answer_context = question_answer_context + f"""\n\n Answer {index+1}: \n{answer}"""

    return question_answer_context


def retrieve_response(conversation: str) -> str:
  """
  This function takes in conversation and return most similar conversation from the FAISS vectorstore
  """
  unique_docs = vectorstore.similarity_search(conversation)
  
  qa_pair = generation_qa_pair(unique_docs)

  return qa_pair

def generate_response(question: str, qa_pair : str,config_id : str) -> str: 
  """
  This function generate soap notes taking a conversation, 
  qa_pair as context for format and config_id for chat_history management.
  """
  
  response_rag = with_message_history_rag.invoke(
    {"messages": [HumanMessage(content=f"""Write an answer for the asked question: \n {question}.Here is an example of a question and respective multiple answers answered by humans: \n {qa_pair}""")]
     },
     config={"configurable": {"session_id": config_id}}
    )
  
  return response_rag.content

def chat(prompt : str ,config_id : str) -> str:
    """
    This function gives user a functionality to chat with the llm keeping chat history as context (only for rag based generation)
    """
    response_prompt = with_message_history_rag.invoke(
    {"messages": [HumanMessage(content=prompt)]
     },
     config={"configurable": {"session_id": config_id}}
    )
  
    return response_prompt.content

