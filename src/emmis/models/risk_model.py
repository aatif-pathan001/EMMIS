import logging
import os
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from emmis.config import settings

logger = logging.getLogger(__name__)


class RiskScoringModel:
    _MODEL_FILE: str = "risk_model.pkl"
    _SCALER_FILE: str = "risk_scaler.pkl"

    def __init__(self) -> None:
        self.model_dir = settings.MODEL_DIR
        self._model_path = os.path.join(self.model_dir, self._MODEL_FILE)
        self._scaler_path = os.path.join(self.model_dir, self._SCALER_FILE)
        self.model: RandomForestClassifier
        self.scaler: MinMaxScaler = MinMaxScaler()
        self._initialise()

    def _generate_training_data(self, n_samples: int = 2000) -> tuple:
        """
        Generate synthetic data to train the model.
        """
        rng = np.random.default_rng(42)

        nlp_risk = rng.beta(2, 3, n_samples)
        sentiment = rng.beta(2, 3, n_samples)
        anomaly = rng.beta(1.5, 4, n_samples)
        kw_counts = rng.poisson(2, n_samples).astype(float)
        anom_regions = rng.poisson(1.5, n_samples).astype(float)

        kw_norm = np.clip(kw_counts / 10.0, 0, 1)
        anom_norm = np.clip(anom_regions / 15.0, 0, 1)

        X = np.column_stack([nlp_risk, sentiment, anomaly, kw_norm, anom_norm])

        composite = (
            0.35 * nlp_risk
            + 0.25 * sentiment
            + 0.25 * anomaly
            + 0.10 * kw_norm
            + 0.05 * anom_norm
        )
        y = np.where(composite > 0.60, 2, np.where(composite > 0.30, 1, 0))

        return X, y

    def _train(self) -> None:
        """Train RandomForest on synthetic data."""
        logger.info("Training risk model on synthetic data ...")
        X, y = self._generate_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_s = self.scaler.fit_transform(X_train)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train_s, y_train)

        X_test_s = self.scaler.transform(X_test)
        acc = self.model.score(X_test_s, y_test)
        logger.info("Risk model training accuracy: %.2f%%", acc * 100)

    def _save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self._model_path)
        joblib.dump(self.scaler, self._scaler_path)
        logger.info("Risk model artefacts saved.")

    def _load(self) -> bool:
        try:
            self.model = joblib.load(self._model_path)
            self.scaler = joblib.load(self._scaler_path)
            logger.info("Risk model artefacts loaded.")
            return True
        except Exception:
            return False

    def _initialise(self) -> None:
        if not self._load():
            self._train()
            self._save()

    def _weighted_score(
        self,
        nlp_risk_score: float,
        sentiment_score: float,
        anomaly_score: float,
        keyword_count: int,
        anomaly_regions: int,
    ) -> float:
        return round(
            min(
                settings.WEIGHTS["nlp_risk_score"] * nlp_risk_score
                + settings.WEIGHTS["sentiment_score"] * sentiment_score
                + settings.WEIGHTS["anomaly_score"] * anomaly_score
                + settings.WEIGHTS["keyword_count"] * min(keyword_count / 10.0, 1.0)
                + settings.WEIGHTS["anomaly_regions"]
                * min(anomaly_regions / 15.0, 1.0),
                1.0,
            ),
            4,
        )

    def predict(
        self,
        nlp_risk_score: float,
        sentiment_score: float,
        anomaly_score: float,
        keyword_count: int,
        anomaly_regions: int,
    ) -> Dict[str, Any]:
        kw_norm = min(keyword_count / 10.0, 1.0)
        ar_norm = min(anomaly_regions / 15.0, 1.0)

        features = np.array(
            [
                [
                    nlp_risk_score,
                    sentiment_score,
                    anomaly_score,
                    kw_norm,
                    ar_norm,
                ]
            ]
        )

        features_scaled = self.scaler.transform(features)
        risk_class = int(self.model.predict(features_scaled)[0])
        proba_raw = self.model.predict_proba(features_scaled)[0]

        proba: Dict[str, float] = {lbl: 0.0 for lbl in settings.RISK_LABELS.values()}
        for cls_idx, cls_label in zip(self.model.classes_, proba_raw):
            proba[settings.RISK_LABELS[int(cls_idx)]] = round(float(cls_label), 4)

        unified = self._weighted_score(
            nlp_risk_score,
            sentiment_score,
            anomaly_score,
            keyword_count,
            anomaly_regions,
        )
        risk_level = settings.RISK_LABELS[risk_class]
        contributions = {
            "nlp_analysis": round(
                settings.WEIGHTS["nlp_risk_score"] * nlp_risk_score, 4
            ),
            "sentiment": round(
                settings.WEIGHTS["sentiment_score"] * sentiment_score, 4
            ),
            "image_anomaly": round(
                settings.WEIGHTS["anomaly_score"] * anomaly_score, 4
            ),
            "risk_keywords": round(settings.WEIGHTS["keyword_count"] * kw_norm, 4),
            "anomaly_regions": round(settings.WEIGHTS["anomaly_regions"] * ar_norm, 4),
        }

        return {
            "unified_risk_score": unified,
            "risk_level": risk_level,
            "risk_probabilities": proba,
            "feature_importance": contributions,
        }
