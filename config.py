"""Shared configuration for index and retriever."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma")
COLLECTION_NAME = "docker_docs_rag"

# OpenRouter (https://openrouter.ai) – API key i base URL z .env
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Modele DeepSeek – format OpenRouter: provider/model
# LLM: deepseek-chat lub deepseek-v3 / deepseek-v3.2
SMART_LLM_MODEL = "deepseek/deepseek-chat"
# Grader (Check & Refine): ocena relewancji chunków – model z lepszą analizą
GRADER_LLM_MODEL = "deepseek/deepseek-v3.2-speciale"
# Embeddingi: OpenAI działa przez OpenRouter. Qwen3-embedding powodował "No embedding data received".
EMBEDDING_MODEL = "openai/text-embedding-3-small"
