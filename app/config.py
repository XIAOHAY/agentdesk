"""全局配置。优先 pydantic-settings；未安装则退化为环境变量/.env（最小依赖）。"""
from __future__ import annotations

import os


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
        openai_api_key: str = ""
        openai_base_url: str = "https://api.openai.com/v1"
        chat_model: str = "gpt-4o-mini"
        embedding_model: str = "text-embedding-3-small"
        top_k: int = 5
        chunk_size: int = 512
        chunk_overlap: int = 64
        max_iterations: int = 3
        use_mcp: bool = False
        vector_backend: str = "memory"
        qdrant_url: str = "http://localhost:6333"
        qdrant_collection: str = "agentdesk"
        redis_url: str = ""
        mem_enabled: bool = True
        mem_collection: str = "agentdesk_memory"
        mem_short_window_k: int = 2
        mem_summarize_every_n: int = 3
        mem_long_top_k: int = 3
        mem_dedup_threshold: float = 0.92
        mem_conflict_threshold: float = 0.80
        mem_event_ttl_days: int = 30
        mem_max_per_user: int = 500
        trace_log: bool = True

        @property
        def use_llm(self) -> bool:
            return bool(self.openai_api_key.strip())

    settings = Settings()

except ImportError:
    _load_env_file()

    class _FallbackSettings:
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        chat_model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
        embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        top_k = int(os.environ.get("TOP_K", "5"))
        chunk_size = int(os.environ.get("CHUNK_SIZE", "512"))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "64"))
        max_iterations = int(os.environ.get("MAX_ITERATIONS", "3"))
        use_mcp = os.environ.get("USE_MCP", "").strip() in ("1", "true", "True")
        vector_backend = os.environ.get("VECTOR_BACKEND", "memory")
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_collection = os.environ.get("QDRANT_COLLECTION", "agentdesk")
        redis_url = os.environ.get("REDIS_URL", "")
        mem_enabled = os.environ.get("MEM_ENABLED", "1").strip() not in ("0", "false", "False")
        mem_collection = os.environ.get("MEM_COLLECTION", "agentdesk_memory")
        mem_short_window_k = int(os.environ.get("MEM_SHORT_WINDOW_K", "2"))
        mem_summarize_every_n = int(os.environ.get("MEM_SUMMARIZE_EVERY_N", "3"))
        mem_long_top_k = int(os.environ.get("MEM_LONG_TOP_K", "3"))
        mem_dedup_threshold = float(os.environ.get("MEM_DEDUP_THRESHOLD", "0.92"))
        mem_conflict_threshold = float(os.environ.get("MEM_CONFLICT_THRESHOLD", "0.80"))
        mem_event_ttl_days = int(os.environ.get("MEM_EVENT_TTL_DAYS", "30"))
        mem_max_per_user = int(os.environ.get("MEM_MAX_PER_USER", "500"))
        trace_log = os.environ.get("TRACE_LOG", "1").strip() not in ("0", "false", "False")

        @property
        def use_llm(self) -> bool:
            return bool(self.openai_api_key.strip())

    settings = _FallbackSettings()
