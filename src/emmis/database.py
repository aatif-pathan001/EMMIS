import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import DESCENDING, MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when a MongoDB operation fails."""

    pass


class MyMongoDBClient:
    _CONNECT_TIMEOUT_MS = 3_000

    def __init__(
        self,
        uri: str,
        db_name: str,
        collection_name: str,
    ) -> None:
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._collection: Optional[Collection] = None
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        try:
            client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=self._CONNECT_TIMEOUT_MS,
            )
            client.admin.command("ping")
            self._collection = client[self.db_name][self.collection_name]
            self._connected = True
            logger.info("Connected to MongoDB.")

        except (ConnectionFailure, ServerSelectionTimeoutError, PyMongoError) as exc:
            self._connected = False
            self._collection = None
            logger.warning("MongoDB unavailable (%s)", exc)

    def _check_connection(self) -> None:
        if not self._connected or self._collection is None:
            raise DatabaseError(
                "MongoDB is unavailable. Check your connection URI and network access."
            )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def insert_analysis(self, data: Dict[str, Any]) -> str:
        self._check_connection()
        assert self._collection is not None
        record: Dict[str, Any] = {
            "_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        try:
            self._collection.insert_one(record)
            return record["_id"]
        except PyMongoError as exc:
            logger.error("Insert failed: %s .", exc)
            raise DatabaseError(f"Failed to insert record: {exc}") from exc

    def get_recent_records(self, limit: int = 10) -> List[Dict[str, Any]]:
        self._check_connection()
        assert self._collection is not None
        try:
            cursor = (
                self._collection.find(
                    {},
                    {
                        "_id": 1,
                        "timestamp": 1,
                        "risk_level": 1,
                        "unified_risk_score": 1,
                        "nlp_risk_score": 1,
                        "anomaly_score": 1,
                        "decrypted_text": 1,
                        "image_analyzed": 1,
                    },
                )
                .sort("timestamp", DESCENDING)
                .limit(limit)
            )
            return [{**doc, "_id": str(doc["_id"])} for doc in cursor]
        except PyMongoError as exc:
            logger.error("Query failed: %s.", exc)
            raise DatabaseError(f"Failed to fetch records: {exc}") from exc

    def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        self._check_connection()
        assert self._collection is not None
        try:
            doc = self._collection.find_one({"_id": record_id})
            if doc:
                return {**doc, "_id": str(doc["_id"])}
            return None
        except PyMongoError as exc:
            logger.error("Failed to find record: %s.", exc)
            raise DatabaseError(f"Failed to fetch records: {exc}") from exc

    def get_total_count(self) -> int:
        if not self._connected or self._collection is None:
            return 0
        try:
            return self._collection.count_documents({})
        except PyMongoError:
            return 0
