import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from emmis.config import settings
from emmis.encryption import Cipher
from emmis.models.language_model import CallSentimentPipeline, TextProcessor
from emmis.models.vision_model import ImageAnomalyDetector
from emmis.models.risk_model import RiskScoringModel
from emmis.database import MyMongoDBClient
from emmis.api.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    EncryptRequest,
    EncryptResponse,
    HealthResponse,
    RecordListResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.cipher = Cipher()
        pipeline = CallSentimentPipeline(model_name=settings.MODEL_NAME)
        app.state.text_processor = TextProcessor(pipeline)
        app.state.image_detector = ImageAnomalyDetector()
        app.state.risk_model = RiskScoringModel()
        app.state.db_client = MyMongoDBClient(
            uri=settings.MONGODB_URI,
            db_name=settings.DATABASE_NAME,
            collection_name=settings.COLLECTION_NAME,
        )

        yield
    except Exception as exc:
        logger.error(f"Initialization failed: {exc}")

    finally:
        logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_analysis(
    encrypted_text: str, request: Request, image_bytes: Optional[bytes] = None
) -> AnalysisResponse:
    # 1. Decrypt
    try:
        decrypted_text = request.app.state.cipher.decrypt_text(encrypted_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {exc}")

    # 2. NLP analysis
    nlp_results = request.app.state.text_processor.process(decrypted_text)

    # 3. CV analysis
    cv_results: dict = {
        "success": False,
        "anomaly_detected": False,
        "anomaly_score": 0.0,
        "anomaly_regions": 0,
    }
    if image_bytes:
        cv_results = request.app.state.image_detector.detect(image_bytes)

    # 4. Risk scoring
    risk_assessment = request.app.state.risk_model.predict(
        nlp_risk_score=nlp_results["nlp_risk_score"],
        sentiment_score=nlp_results["sentiment"]["risk_contribution"],
        anomaly_score=cv_results.get("anomaly_score", 0.0),
        keyword_count=nlp_results["risk_keywords"]["keyword_count"],
        anomaly_regions=cv_results.get("anomaly_regions", 0),
    )

    # 5. Store in db
    record_id = request.app.state.db_client.insert_analysis(
        {
            "decrypted_text": decrypted_text,
            "image_analyzed": image_bytes is not None,
            "nlp_risk_score": nlp_results["nlp_risk_score"],
            "anomaly_score": cv_results.get("anomaly_score", 0.0),
            "unified_risk_score": risk_assessment["unified_risk_score"],
            "risk_level": risk_assessment["risk_level"],
        }
    )

    return AnalysisResponse(
        record_id=record_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decrypted_text=decrypted_text,
        nlp_results=nlp_results,
        cv_results=cv_results,
        risk_assessment=risk_assessment,
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        total_records=request.app.state.db_client.get_total_count(),
    )


@app.post("/api/encrypt", response_model=EncryptResponse, tags=["Encryption"])
async def encrypt_data(input: EncryptRequest, request: Request) -> EncryptResponse:
    try:
        encrypted_text = request.app.state.cipher.encrypt_text(input.text)
        checksum = request.app.state.cipher.checksum(input.text.encode())

        encrypted_image = None
        if input.image_base64:
            raw_img = base64.b64decode(input.image_base64)
            encrypted_image = request.app.state.cipher.encrypt_image(raw_img)

        return EncryptResponse(
            encrypted_text=encrypted_text,
            encrypted_image=encrypted_image,
            text_checksum=checksum,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_json(input: AnalyzeRequest, request: Request) -> AnalysisResponse:
    image_bytes = None
    if input.encrypted_image:
        try:
            image_bytes = request.app.state.cipher.decrypt_image(input.encrypted_image)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Image decryption failed: {exc}"
            )

    try:
        return _run_analysis(input.encrypted_text, request, image_bytes)
    except Exception as exc:
        logger.exception("Analysis pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/records", response_model=RecordListResponse, tags=["Records"])
async def get_records(request: Request, limit: int = 10) -> RecordListResponse:
    limit = min(limit, 50)
    records = request.app.state.db_client.get_recent_records(limit=limit)
    for r in records:
        if "_id" in r and "record_id" not in r:
            r["record_id"] = r.pop("_id")
    return RecordListResponse(
        records=records,
        count=len(records),
    )


@app.get("/api/records/{record_id}", tags=["Records"])
async def get_record(record_id: str, request: Request) -> dict:
    record = request.app.state.db_client.get_record_by_id(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found.")
    if "_id" in record:
        record["record_id"] = record.pop("_id")
    return record
