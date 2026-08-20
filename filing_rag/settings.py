"""Environment and filesystem settings for filing-rag."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "config" / "corpus.yaml"
DEFAULT_CHUNKING_PATH = PROJECT_ROOT / "config" / "chunking.yaml"
DEFAULT_RETRIEVAL_PATH = PROJECT_ROOT / "config" / "retrieval.yaml"
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_set.yaml"
DEFAULT_MIGRATIONS_DIR = PROJECT_ROOT / "migrations"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


class Settings(BaseSettings):
    """Runtime settings loaded from the environment and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    edgar_user_agent: str = ""
    database_url: str = "postgresql://filing:filing@localhost:5432/filing_rag"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    data_dir: Path = PROJECT_ROOT / "data"
    corpus_path: Path = DEFAULT_CORPUS_PATH
    chunking_path: Path = DEFAULT_CHUNKING_PATH
    retrieval_path: Path = DEFAULT_RETRIEVAL_PATH
    golden_path: Path = DEFAULT_GOLDEN_PATH
    migrations_dir: Path = DEFAULT_MIGRATIONS_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    def require_user_agent(self) -> str:
        """Return the EDGAR User-Agent, or raise if it is missing.

        Live fetches must send a name and contact email. The SEC returns 403
        without one: https://www.sec.gov/os/accessing-edgar-data
        """
        agent = self.edgar_user_agent.strip()
        if not agent:
            raise ValueError(
                "EDGAR_USER_AGENT is not set. Copy .env.example to .env and add "
                "a descriptive User-Agent with a contact email."
            )
        return agent

    def require_database_url(self) -> str:
        """Return DATABASE_URL, or raise if it is blank."""
        url = self.database_url.strip()
        if not url:
            raise ValueError(
                "DATABASE_URL is not set. Copy .env.example to .env or start "
                "Postgres with: docker compose up -d"
            )
        return url

    def require_api_key(self) -> str:
        """Return OPENAI_API_KEY, or raise if it is blank."""
        key = self.openai_api_key.strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "an API key for the OpenAI-compatible generation endpoint."
            )
        return key

    def require_base_url(self) -> str:
        """Return OPENAI_BASE_URL, or raise if it is blank."""
        url = self.openai_base_url.strip().rstrip("/")
        if not url:
            raise ValueError(
                "OPENAI_BASE_URL is not set. Copy .env.example to .env or leave "
                "the default https://api.openai.com/v1."
            )
        return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
