"""
create_narrative_training_data.py

Creates a filtered NLP training dataset from processed complaints data.

Input:
data/processed/complaints_processed.parquet

Output:
data/processed/narratives_training.parquet
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


PROCESSED_DATA_PATH = Path(config["paths"]["processed_data_path"])
OUTPUT_PATH = Path(
    config["paths"].get(
        "narratives_training_path",
        "data/processed/narratives_training.parquet",
    )
)

BATCH_SIZE = config["nlp"].get("narrative_batch_size", 50_000)


def create_narrative_training_data() -> None:
    """Create narrative-only training parquet for NLP models."""
    writer = None

    try:
        logger.info("Narrative training data creation started")

        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()

        columns = [
            "Consumer complaint narrative",
            "Product",
            "Issue",
        ]

        dataset = ds.dataset(
            PROCESSED_DATA_PATH,
            format="parquet",
        )

        scanner = dataset.scanner(
            columns=columns,
            filter=(
                ds.field("Consumer complaint narrative").is_valid()
                & ds.field("Product").is_valid()
                & ds.field("Issue").is_valid()
            ),
            batch_size=BATCH_SIZE,
        )

        total_rows = 0

        for batch in scanner.to_batches():
            table = pa.Table.from_batches([batch])

            if table.num_rows == 0:
                continue

            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_PATH,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)
            total_rows += table.num_rows

            logger.info(f"Written rows: {total_rows}")

        if total_rows == 0:
            raise ValueError("No valid narrative rows found for NLP training")

        logger.info(f"Narrative training data saved: {OUTPUT_PATH}")
        logger.info(f"Total narrative rows: {total_rows}")

    except Exception as e:
        logger.exception("Narrative training data creation failed")
        raise CustomException(e, sys)

    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    create_narrative_training_data()