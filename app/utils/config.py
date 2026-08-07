"""
Central configuration for TruthLens AI.
Loads from environment variables / .env file via pydantic-settings.
Import `settings` anywhere in the app instead of calling os.getenv directly.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "huggingface"
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_token: str = ""

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    vector_db: str = "faiss"
    vector_db_path: str = "./data/vector_store"

    # Retrieval
    top_k: int = 5

    # Trust score weights
    weight_retrieval_relevance: float = 0.25
    weight_claim_support: float = 0.35
    weight_citation_coverage: float = 0.20
    weight_model_confidence: float = 0.20

    # Self-correction
    trust_score_threshold: float = 0.7
    max_correction_rounds: int = 2

    # Backend
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Optional external providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""


settings = Settings()
