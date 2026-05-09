from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGODB_URI: str =""
    MODEL_DIR: str=""

    XOR_KEY: int = 42
    SHIFT_VALUE: int = 7
    SCRAMBLE_SEED: int = 13

    HIGH_RISK_THRESHOLD: float = 0.65
    MEDIUM_RISK_THRESHOLD: float = 0.35


settings = Settings()