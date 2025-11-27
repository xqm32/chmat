import json
from glob import glob
from pathlib import Path
from typing import Any

from loguru import logger
from polars import DataFrame

from main import SOURCE_COLLECTION_PREFIX, TARGET_COLLECTION_PREFIX

if __name__ == "__main__":
    logger.add("collect.log")

    results_files = glob("*.results.json")
    logger.info(f"Found results files: {results_files}")

    fn011b03_results: list[dict[str, Any]] = []
    fn030b02_results: list[dict[str, Any]] = []

    for results_file in results_files:
        logger.info(f"Processing file: {results_file}")
        results = json.loads(Path(results_file).read_text())

        for result in results:
            target_prefix = TARGET_COLLECTION_PREFIX
            target_point = result["target_point"]
            source_point = result["source_point"]
            target_id = target_point["id"]
            target_payload = target_point["payload"]

            source_prefix = SOURCE_COLLECTION_PREFIX
            source_id = source_point["id"]
            source_payload = source_point["payload"]
            source_score = source_point["score"]

            if "Fn011b03" in target_payload:
                fn011b03_results.append(
                    {
                        f"{target_prefix}id": target_id,
                        **{f"{target_prefix}{k}": v for k, v in target_payload.items()},
                        f"{source_prefix}id": source_id,
                        **{f"{source_prefix}{k}": v for k, v in source_payload.items()},
                        f"{source_prefix}score": source_score,
                    }
                )
            else:
                fn030b02_results.append(
                    {
                        f"{target_prefix}id": target_id,
                        **{f"{target_prefix}{k}": v for k, v in target_payload.items()},
                        f"{source_prefix}id": source_id,
                        **{f"{source_prefix}{k}": v for k, v in source_payload.items()},
                        f"{source_prefix}score": source_score,
                    }
                )

    logger.info(f"Total Fn011b03 results: {len(fn011b03_results)}")
    logger.info(f"Total Fn030b02 results: {len(fn030b02_results)}")

    fn011b03_df = DataFrame(fn011b03_results, infer_schema_length=None)
    fn011b03_df.write_csv("fn011b03.results.csv")
    fn030b02_df = DataFrame(fn030b02_results, infer_schema_length=None)
    fn030b02_df.write_csv("fn030b02.results.csv")
