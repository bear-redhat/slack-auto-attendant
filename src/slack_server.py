import os
import logging
import re
import pinecone
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
from slack_sdk.oauth.installation_store import FileInstallationStore
from slack_bolt.oauth import OAuthFlow
from slack_bolt.oauth.oauth_settings import OAuthSettings

from conversation import create_conversation

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", '')
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", '')

SLACK_CLIENT_ID = os.environ.get("SLACK_CLIENT_ID")
SLACK_CLIENT_SECRET = os.environ.get("SLACK_CLIENT_SECRET")
SLACK_INSTALLATION_BASE = os.environ.get("SLACK_INSTALLATION_BASE", '/data')

if not SLACK_CLIENT_ID:
    raise ValueError("SLACK_CLIENT_ID environment variable not set.")
if not SLACK_CLIENT_SECRET:
    raise ValueError("SLACK_CLIENT_SECRET environment variable not set.")

# OAuth redirect: slack/oauth_redirect
# Ref: https://slack.dev/bolt-python/concepts#authenticating-oauth
app = AsyncApp(
    oauth_flow=OAuthFlow(
        settings=OAuthSettings(
            scopes=["channels:history", "users:read", "app_mentions:read", "chat:write"],
            installation_store=FileInstallationStore(base_dir=SLACK_INSTALLATION_BASE),  # A simple File based installation store
        )
    )
)

@app.event("app_mention")
async def handle_mention(body, say, logger):
    logger.info(body)

    evt = body.get('event', {})

    channel_id = evt.get('channel')
    thread_ts = evt.get('thread_ts') if 'thread_ts' in evt else evt.get('ts')

    replies = await get_replies(channel_id, thread_ts)
    text = replies[-1].get('text')
    replies = replies[:-1]  # the last one is the one we just received
    resp = await get_ai_response(replies=replies, question=text)

    await say(resp, thread_ts=thread_ts)

async def convert_ids_to_usernames(text):
    user_ids = re.findall('<@(U.*?)>', text)
    for user_id in user_ids:
        username = await get_user_name(user_id)
        text = text.replace(f'<@{user_id}>', f'@{username}')
    return text

async def get_replies(channel_id, thread_ts):
    conversation_replies = await app.client.conversations_replies(channel=channel_id, ts=thread_ts)
    replies = conversation_replies.get('messages', [])
    replies = [
        {
            'user': await get_user_name(r.get('user')),
            'text': await convert_ids_to_usernames(r.get('text')),
        } for r in replies]
    return replies

async def get_user_name(user_id):
    user_info = await app.client.users_info(user=user_id)
    user_name = user_info.get('user', {}).get('name')
    return user_name

async def get_ai_response(replies, question):
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
    )
    system_prompt_str = None
    # FIXME: how to pass the path?
    with open('system_prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt_str = f.read()

    chat_history = []
    for reply in replies:
        chat_history.append((reply['user'], reply['text']))

    conversation = create_conversation(
        system_prompt_str=system_prompt_str,
        chat_history=chat_history
        )
    return conversation(question)

api = AsyncSlackRequestHandler(app)

def start_server(port=9000):
    app.start(port=port, path='/slack')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server()
