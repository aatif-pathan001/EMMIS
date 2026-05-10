import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageAnomalyDetector:
    def __init__(
        self,
        blur_kernel_size: int = 5,
        std_multiplier: float = 2.0,
        min_contour_area: int = 150,
        max_image_dim: int = 800,
        anomaly_threshold: float = 0.20,
    ) -> None:
        k = blur_kernel_size
        self.blur_kernel = (k if k % 2 == 1 else k + 1, k if k % 2 == 1 else k + 1)
        self.std_multiplier = std_multiplier
        self.min_contour_area = min_contour_area
        self.max_image_dim = max_image_dim
        self.anomaly_threshold = anomaly_threshold

    def _decode(self, image_bytes: bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            try:
                from PIL import Image

                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                logger.error("Image decode failed: %s", exc)
                return None
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        largest = max(h, w)
        if largest > self.max_image_dim:
            scale = self.max_image_dim / largest
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return img

    def _build_anomaly_mask(self, gray: np.ndarray) -> Tuple[np.ndarray, float, float]:
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        mean = float(np.mean(blurred))
        std = float(np.std(blurred))
        upper = min(mean + self.std_multiplier * std, 255.0)
        lower = max(mean - self.std_multiplier * std, 0.0)

        mask_bright = (blurred > upper).astype(np.uint8) * 255
        mask_dark = (blurred < lower).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask_bright, mask_dark)
        return mask, mean, std

    def _extract_contours(self, mask: np.ndarray) -> Tuple[List[np.ndarray], float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [
            c for c in contours if cv2.contourArea(c) >= self.min_contour_area
        ]
        total_area = sum(cv2.contourArea(c) for c in significant)
        return significant, total_area

    def _compute_score(
        self,
        anomaly_area: float,
        image_area: float,
        contour_count: int,
    ) -> float:
        if image_area == 0:
            return 0.0

        area_score = min((anomaly_area / image_area) * 15.0, 1.0)
        count_score = min(contour_count / 10.0, 1.0)

        return round(0.70 * area_score + 0.30 * count_score, 4)

    def detect(self, image_bytes: bytes) -> Dict[str, Any]:
        img = self._decode(image_bytes)
        if img is None:
            return {
                "success": False,
                "error": "Could not decode image — unsupported format or corrupt data.",
                "anomaly_detected": False,
                "anomaly_score": 0.0,
                "anomaly_regions": 0,
            }

        img = self._resize(img)
        h, w = img.shape[:2]
        image_area = float(h * w)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask, mean, std = self._build_anomaly_mask(gray)
        contours, anomaly_area = self._extract_contours(mask)
        score = self._compute_score(anomaly_area, image_area, len(contours))

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        regions = []
        for c in sorted_contours:
            x, y, bw, bh = cv2.boundingRect(c)
            regions.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(bw),
                    "height": int(bh),
                    "area_px": int(cv2.contourArea(c)),
                }
            )

        return {
            "success": True,
            "anomaly_detected": score >= self.anomaly_threshold,
            "anomaly_score": score,
            "anomaly_regions": len(contours),
            "significant_regions": regions,
            "image_stats": {
                "width": w,
                "height": h,
                "mean_pixel": round(mean, 2),
                "std_pixel": round(std, 2),
                "anomaly_area_px": int(anomaly_area),
            },
        }

    def annotate(self, image_bytes: bytes) -> Optional[bytes]:
        img = self._decode(image_bytes)
        if img is None:
            return None

        img = self._resize(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask, _, _ = self._build_anomaly_mask(gray)
        contours, _ = self._extract_contours(mask)

        annotated = img.copy()
        cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        result = self.detect(image_bytes)
        score_text = f"Score: {result['anomaly_score']:.2f}"
        cv2.putText(
            annotated,
            score_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )

        success, buffer = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        return buffer.tobytes() if success else None
