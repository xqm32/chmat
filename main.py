import json
from glob import glob
from itertools import batched
from os import environ
from pathlib import Path
from typing import Any
from uuid import uuid7

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from polars import Config, DataFrame, read_excel
from qdrant_client import QdrantClient, models

load_dotenv()

OPENAI_BASE_URL = environ["OPENAI_BASE_URL"]
OPENAI_API_KEY = environ["OPENAI_API_KEY"]
OPENAI_MODEL = environ["OPENAI_MODEL"]
OPENAI_DIMENSIONS = int(environ["OPENAI_DIMENSIONS"])
OPENAI_BATCH_SIZE = int(environ.get("OPENAI_BATCH_SIZE", 1000))

QDRANT_URL = environ["QDRANT_URL"]

SOURCE_SKIP_ROWS = int(environ.get("SOURCE_SKIP_ROWS", 0))
SOURCE_COLLECTION_COLUMN = environ["SOURCE_COLLECTION_COLUMN"]
SOURCE_VALUE_COLUMN = environ["SOURCE_VALUE_COLUMN"]
SOURCE_COLLECTION_PREFIX = environ.get("SOURCE_COLLECTION_PREFIX", "source_")

TARGET_SKIP_ROWS = int(environ.get("TARGET_SKIP_ROWS", 0))
TARGET_COLLECTION_COLUMN = environ["TARGET_COLLECTION_COLUMN"]
TARGET_VALUE_COLUMN = environ["TARGET_VALUE_COLUMN"]
TARGET_COLLECTION_PREFIX = environ.get("TARGET_COLLECTION_PREFIX", "target_")

CHMAT_LOG_FILE = environ.get("CHMAT_LOG_FILE", "chmat.log")

openai = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, timeout=600)


def embed_data_frame(
    data_frame: DataFrame,
    collection_column: str,
    value_column: str,
    collection_prefix: str = "source_",
    error_prefix: str = "",
):
    collection_names = [
        f"{collection_prefix}{collection}"
        for collection in data_frame[collection_column].unique().to_list()
    ]

    for collection_name in collection_names:
        if not qdrant.collection_exists(collection_name):
            logger.info(f"Creating collection: {collection_name}")

            qdrant.create_collection(  # type: ignore
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=OPENAI_DIMENSIONS, distance=models.Distance.COSINE
                ),
            )

    batches = batched(data_frame.iter_rows(named=True), OPENAI_BATCH_SIZE)
    for i, rows in enumerate(batches):
        logger.info(f"Processing batch {i + 1}")

        points: list[dict[str, Any]] = [
            {
                "id": str(uuid7()),
                "payload": row,
            }
            for row in rows
        ]

        try:
            for data in openai.embeddings.create(
                model=OPENAI_MODEL,
                input=[row[value_column] for row in rows],
            ).data:
                points[data.index]["vector"] = data.embedding

            for point in points:
                collection_name = (
                    f"{collection_prefix}{point['payload'][collection_column]}"
                )
                qdrant.upsert(  # type: ignore
                    collection_name=collection_name,
                    points=[point],
                )
        except Exception as e:
            logger.error(f"Error processing batch {i + 1}: {e}")

            Path(f"{error_prefix}{i + 1}.json").write_text(
                json.dumps(
                    {
                        "collection_column": collection_column,
                        "value_column": value_column,
                        "collection_prefix": collection_prefix,
                        "points": points,
                    }
                )
            )


if __name__ == "__main__":
    logger.add(CHMAT_LOG_FILE)

    source_files = glob("*.source.xlsx")
    logger.info(f"Found source files: {source_files}")

    target_files = glob("*.target.xlsx")
    logger.info(f"Found target files: {target_files}")

    for source_file in source_files:
        logger.info(f"Processing source file: {source_file}")

        data_frame = read_excel(
            source_file, read_options={"skip_rows": SOURCE_SKIP_ROWS}
        )
        with Config(tbl_rows=1, tbl_cols=-1):
            logger.info(f"Source data frame: {data_frame}")

        embed_data_frame(
            data_frame,
            SOURCE_COLLECTION_COLUMN,
            SOURCE_VALUE_COLUMN,
            SOURCE_COLLECTION_PREFIX,
            error_prefix=f"{source_file}.",
        )

    for target_file in target_files:
        logger.info(f"Processing target file: {target_file}")

        data_frame = read_excel(
            target_file, read_options={"skip_rows": TARGET_SKIP_ROWS}
        )
        with Config(tbl_rows=1, tbl_cols=-1):
            logger.info(f"Target data frame: {data_frame}")

        embed_data_frame(
            data_frame,
            TARGET_COLLECTION_COLUMN,
            TARGET_VALUE_COLUMN,
            TARGET_COLLECTION_PREFIX,
            error_prefix=f"{target_file}.",
        )
