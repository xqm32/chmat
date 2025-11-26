import json
from glob import glob
from pathlib import Path

from loguru import logger

from main import OPENAI_MODEL, openai, qdrant

if __name__ == "__main__":
    logger.add("retry.log")

    error_files = glob("*.xlsx.*.json")
    logger.info(f"Found error files: {error_files}")

    for error_file in error_files:
        logger.info(f"Processing error file: {error_file}")

        error_data = json.loads(Path(error_file).read_text())
        collection_column = error_data["collection_column"]
        value_column = error_data["value_column"]
        collection_prefix = error_data["collection_prefix"]
        points = error_data["points"]

        try:
            for data in openai.embeddings.create(
                model=OPENAI_MODEL,
                input=[point["payload"][value_column] for point in points],
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
            logger.error(f"Error processing file {error_file}: {e}")
            continue

        Path(error_file).unlink()
        logger.info(f"Successfully processed and removed error file: {error_file}")
