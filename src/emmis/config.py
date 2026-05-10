from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List


class Settings(BaseSettings):
    MONGODB_URI: str = ""
    DATABASE_NAME: str = ""
    COLLECTION_NAME: str = ""
    model_config = SettingsConfigDict(env_file=".env")
    MODEL_DIR: str = "model/artifacts"

    MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"

    XOR_KEY: int = 999
    SHIFT_VALUE: int = 999
    SCRAMBLE_SEED: int = 999

    HIGH_RISK_THRESHOLD: float = 0.65
    MEDIUM_RISK_THRESHOLD: float = 0.35

    RISK_KEYWORDS: List[str] = [
        "critical",
        "failure",
        "error",
        "warning",
        "danger",
        "malfunction",
        "breach",
        "anomaly",
        "threat",
        "alert",
        "emergency",
        "fault",
        "defect",
        "abnormal",
        "severe",
        "catastrophic",
        "hazard",
        "compromise",
        "attack",
        "intrusion",
        "overheating",
        "leak",
        "explosion",
        "crash",
        "corrupted",
        "unauthorized",
        "suspicious",
        "damage",
        "risk",
        "unstable",
        "shutdown",
        "offline",
        "down",
        "broken",
        "failed",
        "unavailable",
        "degraded",
        "compromised",
        "insecure",
        "vulnerable",
    ]

    ENTITY_PATTERNS: Dict[str, List[str]] = {
        "EQUIPMENT": [
            r"unit\s+\d+",
            r"machine\s+\d+",
            r"module\s+\d+",
            r"sensor\s+\d+",
            r"node\s+\d+",
        ],
        "LOCATION": [
            r"sector\s+\w+",
            r"zone\s+\w+",
            r"area\s+\w+",
            r"floor\s+\d+",
            r"building\s+\w+",
        ],
        "SEVERITY": [
            r"level\s+\d+",
            r"stage\s+\d+",
            r"grade\s+\w+",
            r"priority\s+\w+",
        ],
    }

    RISK_LABELS: Dict[int, str] = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    WEIGHTS: Dict[str, float] = {
        "nlp_risk_score": 0.35,
        "sentiment_score": 0.25,
        "anomaly_score": 0.25,
        "keyword_count": 0.10,
        "anomaly_regions": 0.05,
    }


settings = Settings()
