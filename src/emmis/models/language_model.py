import re
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import nltk
from emmis.config import settings
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

for resource in ("vader_lexicon", "punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else resource)
    except LookupError:
        nltk.download(resource, quiet=True)


class TextProcessor:
    _WEIGHT_SENTIMENT: float = 0.55
    _WEIGHT_KEYWORDS: float = 0.45

    def __init__(self, sentiment_pipeline: hf_pipeline) -> None:
        self._transformer_pipeline = None
        self._vader_analyzer = None
        self.sentiment_pipeline = sentiment_pipeline

    @property
    def _vader(self):
        if self._vader_analyzer is None:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            self._vader_analyzer = SentimentIntensityAnalyzer()
        return self._vader_analyzer

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        pipeline = self.sentiment_pipeline
        if pipeline is not None:
            result = pipeline(text[:512])[0]
            label: str = result["label"]
            confidence: float = result["score"]
        else:
            scores = self._vader.polarity_scores(text)
            compound = scores["compound"]
            label = "NEGATIVE" if compound < 0 else "POSITIVE"
            confidence = abs(compound) if compound != 0 else 0.5
        risk_contribution = confidence if label == "NEGATIVE" else (1.0 - confidence)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "risk_contribution": round(risk_contribution, 4),
        }

    def detect_risk_keywords(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        found: List[str] = [kw for kw in settings.RISK_KEYWORDS if kw in text_lower]
        score = min(len(found) / 5.0, 1.0)
        return {
            "found_keywords": found,
            "keyword_count": len(found),
            "keyword_risk_score": round(score, 4),
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        result: Dict[str, List[str]] = {}

        for entity_type, patterns in settings.ENTITY_PATTERNS.items():
            found: List[str] = []
            for pat in patterns:
                matches = re.findall(pat, text_lower)
                found.extend(matches)
            if found:
                result[entity_type] = list(dict.fromkeys(found))
        return result

    def _composite_risk_score(
        self,
        sentiment: Dict[str, Any],
        keywords: Dict[str, Any],
    ) -> float:
        score = (
            self._WEIGHT_SENTIMENT * sentiment["risk_contribution"]
            + self._WEIGHT_KEYWORDS * keywords["keyword_risk_score"]
        )
        return round(min(score, 1.0), 4)

    def process(self, text: str) -> Dict[str, Any]:
        sentiment = self.analyze_sentiment(text)
        keywords = self.detect_risk_keywords(text)
        entities = self.extract_entities(text)
        nlp_risk_score = self._composite_risk_score(sentiment, keywords)

        return {
            "original_text": text,
            "sentiment": sentiment,
            "risk_keywords": keywords,
            "entities": entities,
            "nlp_risk_score": nlp_risk_score,
        }


def CallSentimentPipeline(model_name: str) -> hf_pipeline:
    transformer_pipeline = hf_pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        max_length=512,
    )
    logger.info("Loaded DistilBERT sentiment model.")
    return transformer_pipeline


def format_timestamp(iso_str: Optional[str] = None) -> str:
    if iso_str:
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d  %H:%M:%S UTC")
        except ValueError:
            return iso_str
    return datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")


def truncate(text: str, max_length: int = 80) -> str:
    return text if len(text) <= max_length else text[:max_length] + " …"
