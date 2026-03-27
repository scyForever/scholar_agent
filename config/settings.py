from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class AppSettings:
    project_name: str = "ScholarAgent"
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    prompt_dir: Path = BASE_DIR / "data" / "prompts"
    memory_dir: Path = BASE_DIR / "data" / "memory"
    feedback_dir: Path = BASE_DIR / "data" / "feedback"
    evaluation_dir: Path = BASE_DIR / "data" / "evaluation"
    cache_dir: Path = BASE_DIR / "cache"
    log_dir: Path = BASE_DIR / "logs"
    report_dir: Path = BASE_DIR / "reports"
    vector_db_dir: Path = BASE_DIR / "data" / "vector_db"
    whitelist_path: Path = BASE_DIR / "data" / "whitelist.json"
    trace_dir: Path = BASE_DIR / "logs" / "traces"
    memory_db_path: Path = BASE_DIR / "data" / "memory" / "memory.db"
    feedback_path: Path = BASE_DIR / "data" / "feedback" / "feedback.jsonl"
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "180"))
    llm_max_retries: int = 2
    llm_failure_threshold: int = 3
    llm_recovery_time: int = 300
    llm_long_output_max_tokens: int = 8192
    default_fast_mode: bool = False
    max_search_results: int = 20
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_rrf_k: int = 60
    rag_cc_alpha: float = 0.6
    rag_top_k: int = 10
    rag_parallel_workers: int = int(os.getenv("RAG_PARALLEL_WORKERS", "8"))
    vector_collection_name: str = os.getenv("RAG_VECTOR_COLLECTION", "rag_chunks")
    bge_m3_model_path: str = os.getenv(
        "BGE_M3_MODEL_PATH", "/media/a1/16T/lcy/models/bge-m3"
    )
    bge_m3_batch_size: int = int(os.getenv("BGE_M3_BATCH_SIZE", "8"))
    bge_m3_max_length: int = int(os.getenv("BGE_M3_MAX_LENGTH", "1024"))
    bge_m3_use_fp16: bool = os.getenv("BGE_M3_USE_FP16", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    bge_reranker_model_path: str = os.getenv(
        "BGE_RERANKER_MODEL_PATH", "/media/a1/16T/lcy/models/bge-reranker-v2-m3"
    )
    bge_reranker_batch_size: int = int(os.getenv("BGE_RERANKER_BATCH_SIZE", "16"))
    bge_reranker_use_fp16: bool = os.getenv(
        "BGE_RERANKER_USE_FP16", "false"
    ).lower() in {"1", "true", "yes", "on"}
    wos_documents_url: str = os.getenv(
        "WOS_DOCUMENTS_URL", "https://api.clarivate.com/apis/wos-starter/v1/documents"
    )
    wos_default_database: str = os.getenv("WOS_DATABASE", "WOS")
    provider_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.prompt_dir,
            self.memory_dir,
            self.feedback_dir,
            self.evaluation_dir,
            self.cache_dir,
            self.log_dir,
            self.report_dir,
            self.vector_db_dir,
            self.trace_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _provider_defaults() -> Dict[str, Dict[str, Any]]:
    return {
        "scnet": {
            "priority": 4,
            "model": os.getenv("SCNET_MODEL", "MiniMax-M2.5"),
            "base_url": os.getenv(
                "SCNET_BASE_URL", "https://api.scnet.cn/api/llm/v1/chat/completions"
            ),
            "api_key_name": "SCNET_API_KEY",
        },
        "siliconflow": {
            "priority": 3,
            "model": os.getenv("SILICONFLOW_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2"),
            "base_url": os.getenv(
                "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/chat/completions"
            ),
            "api_key_name": "SILICONFLOW_API_KEY",
        },
        "zhipu": {
            "priority": 3,
            "model": os.getenv("ZHIPU_MODEL", "glm-4.7"),
            "base_url": os.getenv(
                "ZHIPU_BASE_URL",
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            ),
            "api_key_name": "ZHIPU_API_KEY",
        },
        "dashscope": {
            "priority": 2,
            "model": os.getenv("DASHSCOPE_MODEL", "qwen-turbo"),
            "base_url": os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            ),
            "api_key_name": "DASHSCOPE_API_KEY",
        },
        "deepseek": {
            "priority": 2,
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
            "base_url": os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"
            ),
            "api_key_name": "DEEPSEEK_API_KEY",
        },
        "moonshot": {
            "priority": 1,
            "model": os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k"),
            "base_url": os.getenv(
                "MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1/chat/completions"
            ),
            "api_key_name": "MOONSHOT_API_KEY",
        },
        "baidu": {
            "priority": 2,
            "model": os.getenv("BAIDU_MODEL", "ERNIE-Speed"),
            "base_url": os.getenv("BAIDU_BASE_URL", ""),
            "api_key_name": "BAIDU_API_KEY",
        },
        "tencent": {
            "priority": 1,
            "model": os.getenv("TENCENT_MODEL", "hunyuan-lite"),
            "base_url": os.getenv("TENCENT_BASE_URL", ""),
            "api_key_name": "TENCENT_API_KEY",
        },
        "spark": {
            "priority": 1,
            "model": os.getenv("SPARK_MODEL", "spark-lite"),
            "base_url": os.getenv("SPARK_BASE_URL", ""),
            "api_key_name": "SPARK_API_KEY",
        },
        "doubao": {
            "priority": 1,
            "model": os.getenv("DOUBAO_MODEL", "doubao-lite"),
            "base_url": os.getenv("DOUBAO_BASE_URL", ""),
            "api_key_name": "DOUBAO_API_KEY",
        },
        "baichuan": {
            "priority": 1,
            "model": os.getenv("BAICHUAN_MODEL", "Baichuan2"),
            "base_url": os.getenv("BAICHUAN_BASE_URL", ""),
            "api_key_name": "BAICHUAN_API_KEY",
        },
    }


settings = AppSettings(provider_configs=_provider_defaults())
settings.ensure_runtime_dirs()
