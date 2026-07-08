import json
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_edge import Query, QueryRequest, ScrollRequest

from main import (
    OPENAI_BATCH_SIZE,
    SOURCE_COLLECTION_PREFIX,
    TARGET_COLLECTION_PREFIX,
    collections,
)

if __name__ == "__main__":
    logger.add("search.log")

    source_collections = [
        collection
        for collection in collections.keys()
        if collection.startswith(SOURCE_COLLECTION_PREFIX)
    ]
    target_collections = [
        collection
        for collection in collections.keys()
        if collection.startswith(TARGET_COLLECTION_PREFIX)
    ]
    logger.info(f"Source collections: {source_collections}")
    logger.info(f"Target collections: {target_collections}")

    for target_collection in target_collections:
        target_collection_name = target_collection
        collection_name = target_collection_name.removeprefix(TARGET_COLLECTION_PREFIX)
        source_collection_name = f"{SOURCE_COLLECTION_PREFIX}{collection_name}"
        logger.info(
            f"Processing target collection: {target_collection_name} with source collection: {source_collection_name}"
        )

        if source_collection_name not in collections:
            logger.warning(
                f"Source collection {source_collection_name} does not exist for target collection {target_collection_name}. Skipping."
            )
            continue

        results: list[dict[str, Any]] = []

        offset = None
        while True:
            target_points, offset = collections[target_collection_name].scroll(
                ScrollRequest(
                    limit=OPENAI_BATCH_SIZE,
                    offset=offset,
                    with_vector=True,
                )
            )
            logger.info(
                f"Fetched {len(target_points)} points from target collection: {target_collection_name}"
            )
            logger.info(f"Offset for next scroll: {offset}")

            for target_point in target_points:
                vector = target_point.vector
                if vector is None:
                    logger.warning(
                        f"Target point {target_point.id} in collection {target_collection_name} has no vector. Skipping."
                    )
                    continue
                if isinstance(vector, dict):
                    logger.warning(
                        f"Target point {target_point.id} in collection {target_collection_name} has a vector of type dict. Skipping."
                    )
                    continue

                source_point, *_ = collections[source_collection_name].query(
                    QueryRequest(
                        query=Query.Nearest(vector),
                        limit=1,
                        with_payload=True,
                    )
                )

                results.append(
                    {
                        "target_point": {
                            "id": str(target_point.id),
                            "payload": target_point.payload,
                        },
                        "source_point": {
                            "id": str(source_point.id),
                            "payload": source_point.payload,
                            "score": source_point.score,
                        },
                    }
                )

            if offset is None:
                break

        Path(f"{target_collection_name}.results.json").write_text(
            json.dumps(results, ensure_ascii=False)
        )
        logger.info(f"Wrote results to {target_collection_name}.results.json")
