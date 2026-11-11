"""Load generated Parquet datasets and display Spark sales analytics."""

import logging
import time
from pathlib import Path

from pyspark.sql import DataFrame

from src import config
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def _load_timed(
    analytics: SalesAnalytics,
    dataset_name: str,
    parquet_path: Path,
) -> DataFrame:
    """Load one Parquet dataset and print its load duration."""
    started_at = time.perf_counter()
    dataframe = analytics.load_parquet(parquet_path)
    print(f"Load {dataset_name}: {time.perf_counter() - started_at:.2f} seconds")
    return dataframe


def run_analytics(input_dir: Path = config.RAW_DATA_DIR) -> None:
    """Run and display all sales analytics for Parquet files in ``input_dir``."""
    analytics = SalesAnalytics()
    spark = analytics.create_spark_session()
    try:
        customers = _load_timed(
            analytics,
            "customers",
            input_dir / "customers.parquet",
        )
        products = _load_timed(
            analytics,
            "products",
            input_dir / "products.parquet",
        )
        orders = _load_timed(
            analytics,
            "orders",
            input_dir / "orders.parquet",
        )
        LOGGER.info(
            "Loaded customers (%d columns), products (%d columns), and orders (%d columns)",
            len(customers.columns),
            len(products.columns),
            len(orders.columns),
        )

        operations = (
            (
                "Top customers by revenue",
                lambda: analytics.top_customers_by_revenue(orders, products),
            ),
            (
                "Sales by category",
                lambda: analytics.sales_by_category(orders, products),
            ),
            (
                "Monthly trends",
                lambda: analytics.monthly_trends(orders, products),
            ),
        )
        for operation_name, operation in operations:
            print(f"\n{operation_name}:")
            started_at = time.perf_counter()
            operation().show(truncate=False)
            print(
                f"{operation_name} execution time: "
                f"{time.perf_counter() - started_at:.2f} seconds"
            )
    finally:
        spark.stop()
        analytics.spark = None


def main() -> int:
    """Run the analytics job and return a nonzero status on failure."""
    config.configure_logging()
    try:
        run_analytics()
    except Exception:
        LOGGER.exception("Analytics run failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
