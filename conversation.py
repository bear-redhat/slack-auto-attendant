import os
import logging
from typing import Dict, List, Optional, Tuple
from langchain import OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone as PineconeStore
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


logger = logging.getLogger(__name__)

MAX_TOKEN_SIZE = 4096
MAX_RESPONSE_SIZE = 512
MAX_INPUT_SIZE = MAX_TOKEN_SIZE - MAX_RESPONSE_SIZE
MY_USERNAME = 'little_bear'

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", '')
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", '')
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")


def create_conversation(system_prompt_str: Optional[str]=None, chat_history: Optional[List[Tuple[str, str]]]=None):
    system_prompt_qa = None
    if system_prompt_str:
        system_prompt_qa = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_str),
            HumanMessagePromptTemplate.from_template("{question}"),
            ])

    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_key=OPENAI_API_KEY,
    ) # type: ignore
    vector_store = PineconeStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        )
    model = ChatOpenAI(
        temperature=0,
        model='gpt-3.5-turbo',
        verbose=True,
        ) # type: ignore

    memory = ConversationSummaryBufferMemory(
        llm=model,
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer',
        )

    if chat_history:
        for name, message in chat_history:
            if (name == MY_USERNAME):
                memory.chat_memory.add_ai_message(message)
            else:
                memory.chat_memory.add_user_message(message)

    # doc_chain = load_qa_with_sources_chain(
    #     model,
    #     chain_type="stuff",
    #     verbose=True,
    #     prompt=system_prompt_qa
    #     )

    qa_chain = load_qa_chain(
        model,
        chain_type="stuff",
        verbose=True,
        prompt=system_prompt_qa
    )

    question_generator = LLMChain(
        llm=model,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True,
        )

    conversation = ConversationalRetrievalChain(
        # model,
        question_generator=NoOpLLMChain(),
        retriever=vector_store.as_retriever(),
        combine_docs_chain=qa_chain,
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        max_tokens_limit=MAX_TOKEN_SIZE,
        verbose=True,
        # combine_docs_chain_kwargs={
        #     'prompt': system_prompt,
        # }
        )

    def conversation_wrapper(user_input: str) -> str:
        resp = conversation({"question": user_input})
        answer = resp.get('answer', '') or ''
        sources = resp.get('source_documents', []) or []
        source_paths = [s.metadata.get('source') for s in sources if s.metadata.get('source')]
        # don't return sources. No matter what you asked,
        # even totally irrelevant questions, it will return some random docs
        return answer

    return conversation_wrapper

class NoOpLLMChain(LLMChain):
   """No-op LLM chain."""

   def __init__(self):
       """Initialize."""
       super().__init__(llm=OpenAI(), prompt=PromptTemplate(template="", input_variables=[]))

   async def arun(self, question: str, *args, **kwargs) -> str:
       return question

   def run(self, question, chat_history, callbacks):
       return question
