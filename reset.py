from shutil import rmtree

from loguru import logger

from main import collections

logger.info(f"Existing collections: {collections.keys()}")

for collection_name, collection in collections.items():
    collection.close()

    logger.info(f"Deleting collection: {collection_name}")

    rmtree(f"collections/{collection_name}")
