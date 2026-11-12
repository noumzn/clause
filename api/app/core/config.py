from pathlib import Path
from functools import lru_cache

from langchain_qdrant import init_embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

ENV_FILE=Path(__file__).resolve().parents[2]/".env"


class Settings(BaseSettings):

    model_config=SettingsConfigDict(env_prefix="RAG", env_file="ignore")

    #APP
    app_name:str= "Clause"
    data_path: str="data/shop_handbook.pdf"

    #CHAT
    chat_provider: str="ollama"
    chat_model: str="gemma3:1b"
    temperature: float=0.1
    
    #EMBEDDING
    embed_provider: str="ollama"
    embed_model: str= "nomic=embed-text:latest"
    embed_dim: int=768

    #INDEXING
    chunk_size:int=600
    chunk_overlap:int=100

    #QDRANT
    collection_name: str="shop_handbook"
    qdrant_data="./qdrant_data"                                        #qdrant data path

    #RETRIEVAL
    top_k:int=3
    score_threshold:float=0.5

    #PROMPT
    system: str=(
        "You are a helpful AI Clause shop assistant!!"
        "Answer only from provided context."
    )

@lru_cache
def get_settings():
        return Settings()

def get_embeddings(settings:Settings):
      return init_embeddings(
            model=settings.embed_model,
            provider=settings.embed_provider
            )

def get_qdrant_client(settings:Settings):
      return QdrantClient(
            path=settings.qdrant_data
      )


