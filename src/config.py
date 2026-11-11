"""Project paths and logging configuration."""

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CUSTOMERS_FILENAME = "customers.csv"
PRODUCTS_FILENAME = "products.csv"
ORDERS_FILENAME = "orders.csv"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging for project scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
