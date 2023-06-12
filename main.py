import os
import argparse
import logging
from typing import  Optional
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone as PineconeStore
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
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

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", '')
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", '')
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")


def ingest_directory(directory, index_name):
    # Get Pinecone and OpenAI API keys from environment variables
    TEXT_SPLITTER_CHUNK_SIZE = os.environ.get("TEXT_SPLITTER_CHUNK_SIZE", 1000)
    TEXT_SPLITTER_CHUNK_OVERLAP = os.environ.get("TEXT_SPLITTER_CHUNK_OVERLAP", 300)


    logger.info('Loading documents from directory: %s', directory)
    loader = DirectoryLoader(directory, glob='**/*.md',
        show_progress=True)
    raw_docs = loader.load()
    logger.info('Loaded %d documents', len(raw_docs))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap = TEXT_SPLITTER_CHUNK_OVERLAP,
    )

    docs = text_splitter.split_documents(raw_docs)
    logger.info('Split documents into %d chunks', len(docs))

    logger.info('Updating vector store...')
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_key=OPENAI_API_KEY,
    ) # type: ignore

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
    )

    if index_name not in pinecone.list_indexes():
        logger.info('Creating new index: %s', index_name)
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536,     # text-embedding-ada-002 -> 1536 dimensions
        )
        logger.info('Index created')

    logger.info('Removing existing documents from index...')
    index = pinecone.Index(index_name)
    index.delete(delete_all=True)

    logger.info('Adding documents to index...')
    PineconeStore.from_documents(docs, embeddings,
        index_name=index_name,
        )
    logger.info('Done!')

def create_conversation(system_prompt_str: Optional[str]=None):
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

    doc_chain = load_qa_with_sources_chain(
        model,
        chain_type="map_reduce"
        )
    system_prompt = None
    if system_prompt_str:
        system_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_str),
            HumanMessagePromptTemplate.from_template("{question}"),
            ])
    question_generator = LLMChain(
        llm=model,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True,
        )

    conversation = ConversationalRetrievalChain.from_llm(
        model,
        # question_generator=question_generator,
        retriever=vector_store.as_retriever(),
        # combine_docs_chain=doc_chain,
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        max_tokens_limit=MAX_TOKEN_SIZE,
        verbose=True,
        combine_docs_chain_kwargs={
            'prompt': system_prompt,
        }
        )

    def conversation_wrapper(user_input: str) -> str:
        resp = conversation({"question": user_input})
        answer = resp.get('answer', '') or ''
        sources = resp.get('source_documents', []) or []
        return answer

    return conversation_wrapper

def interactive_chat(system_prompt_path: Optional[str]=None):
    """
    This function runs an interactive chat session with the user. It prompts the user for input, processes the input,
    generates a response, and prints the response to the console. The chat session continues until the user enters an
    empty string.
    """
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
    )

    system_prompt_str = None
    if system_prompt_path:
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt_str = f.read()

    conversation = create_conversation(system_prompt_str)
    user_input = ''

    while True:
        try:
            # Read input until Ctrl-D is pressed
            user_input += input("You: ") + "\n"
        except EOFError:
            if not user_input.strip():
                break

            # Process user input and generate response
            response = conversation(user_input)
            print("Bot: " + response.replace("\n", "\n> "))
            user_input = ''  # Reset user input

    print("Goodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenShift CI Slack Auto Attendant')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ingest', type=str, help='The directory containing the .md files to ingest.')
    group.add_argument('--interactive', action='store_true', help='Run in interactive mode.')
    parser.add_argument('--system-prompt', type=str, help='The path to the text file containing the system prompt.', required=False)
    args = parser.parse_args()

    # Check for API keys
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    if not PINECONE_INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME environment variable not set.")
    if not PINECONE_ENVIRONMENT:
        raise ValueError("PINECONE_ENVIRONMENT environment variable not set.")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    if args.interactive and not args.system_prompt:
        raise ValueError("The --system-prompt parameter is required when using --interactive.")

    if args.system_prompt and not args.interactive:
        raise ValueError("The --system-prompt parameter can only be used with --interactive.")

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    if args.ingest:
        ingest_directory(args.ingest, PINECONE_INDEX_NAME)

    if args.interactive:
        interactive_chat(args.system_prompt)
