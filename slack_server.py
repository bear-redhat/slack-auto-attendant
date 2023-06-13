import logging
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler

app = AsyncApp()

@app.event("app_mention")
async def handle_app_mentions(body, say, logger):
    logger.info(body)
    await say("What's up?")

@app.event("message")
async def handle_message(body, say, logger):
    logger.info(body)
    evt = body.get('event', {})

    thread_ts = evt.get('thread_ts') if 'thread_ts' in evt else evt.get('ts')
    text = evt.get('text')

    user_id = evt.get('user')
    user_info = await app.client.users_info(user=user_id)
    user_name = user_info.get('user', {}).get('name')

    channel_id = evt.get('channel')
    conversation_replies = await app.client.conversations_replies(channel=channel_id, ts=thread_ts)
    replies = conversation_replies.get('messages', [])
    replies = [
        {
            'user': await get_user_name(r.get('user')),
            'text': r.get('text'),
        } for r in replies]

    await say(f"Hi {user_name}. You just said {text}\n right?", thread_ts=thread_ts)

async def get_user_name(user_id):
    user_info = await app.client.users_info(user=user_id)
    user_name = user_info.get('user', {}).get('name')
    return user_name

api = AsyncSlackRequestHandler(app)

def start_server(port=9000):
    app.start(port=port, path='/slack')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server()
