from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class EncryptRequest(BaseModel):
    text: str
    image_base64: Optional[str] = None


class EncryptResponse(BaseModel):
    encrypted_text: str
    encrypted_image: Optional[str] = None
    text_checksum: str


class AnalyzeRequest(BaseModel):
    encrypted_text: str
    encrypted_image: Optional[str] = None


class AnalysisResponse(BaseModel):
    record_id: str
    timestamp: str
    decrypted_text: str
    nlp_results: Dict[str, Any]
    cv_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]


class RecordListResponse(BaseModel):
    records: List[Dict[str, Any]]
    count: int
    storage_backend: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    total_records: int
    version: str = "1.0.0"
