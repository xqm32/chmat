import subprocess

from loguru import logger

from main import QDRANT_URL, qdrant

if __name__ == "__main__":
    logger.add("snapshot.log")

    collections = qdrant.get_collections().collections
    logger.info(f"Found {len(collections)} collections.")

    for collection in collections:
        collection_name = collection.name
        logger.info(f"Creating snapshot for collection: {collection_name}")

        snapshot_response = qdrant.create_snapshot(collection_name)
        if not snapshot_response:
            logger.error(f"Failed to create snapshot for collection: {collection_name}")
            continue

        snapshot_name = snapshot_response.name
        logger.info(f"Snapshot created: {snapshot_name}")

        subprocess.run(
            [
                "curl",
                "--remote-name",
                f"{QDRANT_URL}/collections/{collection_name}/snapshots/{snapshot_name}",
            ]
        )
