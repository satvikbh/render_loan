# utils.py
import json
import logging
import os

logger = logging.getLogger(__name__)

CHAT_HISTORY_PATH = os.path.join('conversation_history.json')

def load_chat_history():
    logger.info(f"Loading chat history from {CHAT_HISTORY_PATH}...")
    if not os.path.exists(CHAT_HISTORY_PATH):
        logger.info("No chat history file found, starting with empty history.")
        return {}
    try:
        with open(CHAT_HISTORY_PATH, 'r') as f:
            history = json.load(f)
        logger.info("Chat history loaded successfully.")
        return history
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
        return {}