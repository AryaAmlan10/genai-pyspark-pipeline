"""Generate e-commerce data and save the datasets as Parquet files."""

import argparse
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

from src import config
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)
DEFAULT_CUSTOMER_COUNT = 100_000
DEFAULT_PRODUCT_COUNT = 10_000
DEFAULT_ORDER_COUNT = 1_000_000


def generate_and_save(
    output_dir: Path = config.RAW_DATA_DIR,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    seed: int | None = None,
) -> dict[str, Path]:
    """Generate the three datasets and save each as a Parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_started = perf_counter()
    customers, products, orders = SyntheticDataGenerator(seed=seed).generate_all(
        customer_count=customer_count,
        product_count=product_count,
        order_count=order_count,
    )
    generation_seconds = perf_counter() - generation_started
    print(f"Data generation time: {generation_seconds:.2f} seconds")

    datasets: dict[str, pd.DataFrame] = {
        "customers": customers,
        "products": products,
        "orders": orders,
    }
    output_paths: dict[str, Path] = {}
    for dataset_name, dataframe in datasets.items():
        parquet_path = output_dir / f"{dataset_name}.parquet"
        dataframe.to_parquet(
            parquet_path,
            index=False,
            coerce_timestamps="us",
            allow_truncated_timestamps=True,
        )
        output_paths[dataset_name] = parquet_path
        file_size = parquet_path.stat().st_size
        print(f"{dataset_name}: {file_size:,} bytes ({parquet_path})")

    return output_paths


def _parse_args() -> argparse.Namespace:
    """Parse row counts, output directory, and optional random seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMER_COUNT)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--orders", type=int, default=DEFAULT_ORDER_COUNT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RAW_DATA_DIR,
        help="Directory for generated Parquet files",
    )
    return parser.parse_args()


def main() -> int:
    """Run the generation pipeline and report failures with a nonzero exit code."""
    config.configure_logging()
    args = _parse_args()
    try:
        generate_and_save(
            output_dir=args.output_dir,
            customer_count=args.customers,
            product_count=args.products,
            order_count=args.orders,
            seed=args.seed,
        )
    except Exception:
        LOGGER.exception("Failed to generate and save e-commerce datasets")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
