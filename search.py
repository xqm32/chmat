import json
from pathlib import Path
from typing import Any

from loguru import logger

from main import (
    OPENAI_BATCH_SIZE,
    SOURCE_COLLECTION_PREFIX,
    TARGET_COLLECTION_PREFIX,
    qdrant,
)

if __name__ == "__main__":
    logger.add("search.log")

    collections_response = qdrant.get_collections()
    collections = collections_response.collections

    source_collections = [
        collection
        for collection in collections
        if collection.name.startswith(SOURCE_COLLECTION_PREFIX)
    ]
    target_collections = [
        collection
        for collection in collections
        if collection.name.startswith(TARGET_COLLECTION_PREFIX)
    ]
    logger.info(f"Source collections: {source_collections}")
    logger.info(f"Target collections: {target_collections}")

    for target_collection in target_collections:
        target_collection_name = target_collection.name
        collection_name = target_collection_name.removeprefix(TARGET_COLLECTION_PREFIX)
        source_collection_name = f"{SOURCE_COLLECTION_PREFIX}{collection_name}"
        logger.info(
            f"Processing target collection: {target_collection_name} with source collection: {source_collection_name}"
        )

        if not qdrant.collection_exists(collection_name=source_collection_name):
            logger.warning(
                f"Source collection {source_collection_name} does not exist for target collection {target_collection_name}. Skipping."
            )
            continue

        results: list[dict[str, Any]] = []

        offset = None
        while True:
            target_points, offset = qdrant.scroll(  # type: ignore
                collection_name=target_collection_name,
                limit=OPENAI_BATCH_SIZE,
                offset=offset,
                with_vectors=True,
            )
            logger.info(
                f"Fetched {len(target_points)} points from target collection: {target_collection_name}"
            )
            logger.info(f"Offset for next scroll: {offset}")

            for target_point in target_points:
                query_response = qdrant.query_points(  # type: ignore
                    collection_name=source_collection_name,
                    query=target_point.vector,
                    limit=1,
                )
                source_point, *_ = query_response.points

                results.append(
                    {
                        "target_point": {
                            "id": target_point.id,
                            "payload": target_point.payload,
                        },
                        "source_point": {
                            "id": source_point.id,
                            "payload": source_point.payload,
                            "score": source_point.score,
                        },
                    }
                )

            if offset is None:
                break

        Path(f"{target_collection_name}.results.json").write_text(json.dumps(results))
        logger.info(f"Wrote results to {target_collection_name}.results.json")
