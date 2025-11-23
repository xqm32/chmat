from loguru import logger

from main import qdrant

collections = qdrant.get_collections()
logger.info(f"Existing collections: {[col.name for col in collections.collections]}")

for collection in collections.collections:
    logger.info(f"Deleting collection: {collection.name}")
    qdrant.delete_collection(collection_name=collection.name)
