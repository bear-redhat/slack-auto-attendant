import os
import argparse
import logging
from typing import Optional
import logging
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone as PineconeStore

from conversation import create_conversation
from slack_server import start_server

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", '')
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", '')
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

logger = logging.getLogger(__name__)


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
    group.add_argument('--slack', action='store_true', help='Create Flask server')
    parser.add_argument('--system-prompt', type=str,
        help='The path to the text file containing the system prompt.',
        required=False,
        default='./system-prompt.txt')
    parser.add_argument('--port', type=int,
        help='The port to run the Slack server on.',
        required=False,
        default=9000)
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

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    if args.ingest:
        ingest_directory(args.ingest, PINECONE_INDEX_NAME)

    if args.interactive:
        interactive_chat(args.system_prompt)

    if args.slack:
        start_server(args.port)
