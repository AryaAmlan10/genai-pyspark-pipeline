"""Generate synthetic e-commerce data as pandas DataFrames and CSV datasets."""

import argparse
import csv
import logging
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import (
    CUSTOMERS_FILENAME,
    ORDERS_FILENAME,
    PRODUCTS_FILENAME,
    RAW_DATA_DIR,
    configure_logging,
)

LOGGER = logging.getLogger(__name__)
CATEGORIES = ("Electronics", "Home", "Clothing", "Books", "Sports")
ORDER_STATUSES = ("completed", "shipped", "processing", "cancelled")
CUSTOMER_FIELDS = ("customer_id", "first_name", "last_name", "email", "city", "state", "signup_date")
PRODUCT_FIELDS = ("product_id", "product_name", "category", "price")
ORDER_FIELDS = (
    "order_id",
    "customer_id",
    "product_id",
    "quantity",
    "unit_price",
    "order_date",
    "status",
)
PRODUCT_CATEGORIES = ("Electronics", "Clothing", "Home", "Sports", "Books")


class SyntheticDataGenerator:
    """Generate fictional e-commerce entities and orders as pandas DataFrames."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize seeded NumPy and Faker random generators."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.faker = Faker()
        if seed is not None:
            self.faker.seed_instance(seed)

    def generate_customers(self, count: int = 100_000) -> pd.DataFrame:
        """Generate customers with Faker identities and normally distributed ages."""
        if count < 1:
            raise ValueError("count must be at least 1")

        today = np.datetime64(date.today(), "D")
        registration_start = today - np.timedelta64(5 * 365, "D")
        registration_days = int((today - registration_start).astype(int))
        ages = np.clip(self.rng.normal(loc=35, scale=12, size=count), 18, 90)
        customers: list[dict[str, object]] = []
        for index in tqdm(range(count), desc="Generating customers", unit="customer"):
            customers.append(
                {
                    "customer_id": f"C{index + 1:07d}",
                    "name": self.faker.name(),
                    "email": self.faker.unique.email(),
                    "age": int(round(ages[index])),
                    "city": self.faker.city(),
                    "country": self.faker.country(),
                    "registration_date": registration_start
                    + np.timedelta64(
                        int(self.rng.integers(0, registration_days + 1)), "D"
                    ),
                }
            )
        dataframe = pd.DataFrame(
            customers,
            columns=(
                "customer_id",
                "name",
                "email",
                "age",
                "city",
                "country",
                "registration_date",
            ),
        )
        LOGGER.info("Generated %d customers", len(dataframe))
        return dataframe

    def generate_products(self, count: int = 10_000) -> pd.DataFrame:
        """Generate products with prices, stock counts, and customer ratings."""
        if count < 1:
            raise ValueError("count must be at least 1")

        products: list[dict[str, object]] = []
        for index in tqdm(range(count), desc="Generating products", unit="product"):
            products.append(
                {
                    "product_id": f"P{index + 1:07d}",
                    "name": f"{self.faker.word().title()} {self.faker.word().title()}",
                    "category": str(self.rng.choice(PRODUCT_CATEGORIES)),
                    "price": round(float(self.rng.uniform(10, 500)), 2),
                    "stock": int(self.rng.integers(0, 1_001)),
                    "rating": round(float(self.rng.uniform(1, 5)), 1),
                }
            )
        dataframe = pd.DataFrame(
            products,
            columns=("product_id", "name", "category", "price", "stock", "rating"),
        )
        LOGGER.info("Generated %d products", len(dataframe))
        return dataframe

    def generate_orders(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        count: int = 1_000_000,
    ) -> pd.DataFrame:
        """Generate orders with an 80/20 customer split and Pareto-weighted IDs."""
        if count < 1:
            raise ValueError("count must be at least 1")
        if customers.empty:
            raise ValueError("customers must contain at least one customer")
        if products.empty:
            raise ValueError("products must contain at least one product")
        if "customer_id" not in customers or "product_id" not in products:
            raise ValueError("customers and products must include their ID columns")

        customer_ids = customers["customer_id"].to_numpy()
        top_count = max(1, int(np.ceil(len(customer_ids) * 0.2)))
        high_volume_ids = customer_ids[:top_count]
        low_volume_ids = customer_ids[top_count:]
        high_volume_order_count = int(round(count * 0.8))
        low_volume_order_count = count - high_volume_order_count
        customer_order_ids = np.empty(count, dtype=object)

        for start in tqdm(
            range(0, count, 100_000),
            desc="Generating orders",
            unit="100k orders",
        ):
            end = min(start + 100_000, count)
            high_start = min(start, high_volume_order_count)
            high_end = min(end, high_volume_order_count)
            if high_end > high_start:
                weights = self.rng.pareto(a=1.16, size=len(high_volume_ids)) + 1
                probabilities = weights / weights.sum()
                customer_order_ids[high_start:high_end] = self.rng.choice(
                    high_volume_ids,
                    size=high_end - high_start,
                    p=probabilities,
                )

            low_start = max(start, high_volume_order_count)
            low_end = min(end, count)
            if low_end > low_start:
                if len(low_volume_ids) == 0:
                    fallback_ids = high_volume_ids
                else:
                    fallback_ids = low_volume_ids
                weights = self.rng.pareto(a=1.16, size=len(fallback_ids)) + 1
                probabilities = weights / weights.sum()
                customer_order_ids[low_start:low_end] = self.rng.choice(
                    fallback_ids,
                    size=low_end - low_start,
                    p=probabilities,
                )

        today = np.datetime64(date.today(), "D")
        order_start = today - np.timedelta64(2 * 365, "D")
        order_day_count = int((today - order_start).astype(int)) + 1
        order_dates = order_start + self.rng.integers(
            0, order_day_count, size=count, dtype=np.int64
        ).astype("timedelta64[D]")
        dataframe = pd.DataFrame(
            {
                "order_id": [f"O{index + 1:09d}" for index in range(count)],
                "customer_id": customer_order_ids,
                "product_id": self.rng.choice(
                    products["product_id"].to_numpy(), size=count
                ),
                "quantity": self.rng.integers(1, 11, size=count),
                "order_date": order_dates,
            },
            columns=("order_id", "customer_id", "product_id", "quantity", "order_date"),
        )
        LOGGER.info("Generated %d orders", len(dataframe))
        return dataframe

    def generate_all(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate customers, products, and orders using the requested defaults."""
        customers = self.generate_customers(customer_count)
        products = self.generate_products(product_count)
        orders = self.generate_orders(customers, products, order_count)
        return customers, products, orders


def _write_csv(
    file_path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    """Write a sequence of row dictionaries to a CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %d rows to %s", len(rows), file_path)


def generate_dataset(
    output_dir: Path = RAW_DATA_DIR,
    customer_count: int = 1_000,
    product_count: int = 100,
    order_count: int = 5_000,
    seed: int | None = None,
) -> dict[str, Path]:
    """Generate related customer, product, and order CSV files."""
    if customer_count < 1:
        raise ValueError("customer_count must be at least 1")
    if product_count < 1:
        raise ValueError("product_count must be at least 1")
    if order_count < 1:
        raise ValueError("order_count must be at least 1")

    faker = Faker()
    faker.seed_instance(seed)
    rng = random.Random(seed)

    customers = [
        {
            "customer_id": f"C{index:06d}",
            "first_name": faker.first_name(),
            "last_name": faker.last_name(),
            "email": faker.unique.email(),
            "city": faker.city(),
            "state": faker.state_abbr(),
            "signup_date": faker.date_between(
                start_date=date.today() - timedelta(days=730),
                end_date=date.today(),
            ).isoformat(),
        }
        for index in range(1, customer_count + 1)
    ]

    products: list[dict[str, str]] = []
    for index in range(1, product_count + 1):
        price = Decimal(rng.randint(500, 50_000)) / Decimal(100)
        products.append(
            {
                "product_id": f"P{index:06d}",
                "product_name": f"{faker.word().title()} {faker.word().title()}",
                "category": rng.choice(CATEGORIES),
                "price": f"{price:.2f}",
            }
        )

    orders: list[dict[str, str]] = []
    for index in range(1, order_count + 1):
        product = rng.choice(products)
        orders.append(
            {
                "order_id": f"O{index:07d}",
                "customer_id": rng.choice(customers)["customer_id"],
                "product_id": product["product_id"],
                "quantity": str(rng.randint(1, 5)),
                "unit_price": product["price"],
                "order_date": faker.date_between(
                    start_date=date.today() - timedelta(days=365),
                    end_date=date.today(),
                ).isoformat(),
                "status": rng.choice(ORDER_STATUSES),
            }
        )

    output_paths = {
        "customers": output_dir / CUSTOMERS_FILENAME,
        "products": output_dir / PRODUCTS_FILENAME,
        "orders": output_dir / ORDERS_FILENAME,
    }
    _write_csv(output_paths["customers"], CUSTOMER_FIELDS, customers)
    _write_csv(output_paths["products"], PRODUCT_FIELDS, products)
    _write_csv(output_paths["orders"], ORDER_FIELDS, orders)
    return output_paths


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for generating synthetic datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--customers", type=int, default=100_000, help="Number of customers"
    )
    parser.add_argument(
        "--products", type=int, default=10_000, help="Number of products"
    )
    parser.add_argument(
        "--orders", type=int, default=1_000_000, help="Number of orders"
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory for generated CSV files",
    )
    return parser.parse_args()


def main() -> None:
    """Run the data generator using command-line arguments."""
    configure_logging()
    args = _parse_args()
    customers, products, orders = SyntheticDataGenerator(seed=args.seed).generate_all(
        customer_count=args.customers,
        product_count=args.products,
        order_count=args.orders,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    customers.to_csv(args.output_dir / CUSTOMERS_FILENAME, index=False)
    products.to_csv(args.output_dir / PRODUCTS_FILENAME, index=False)
    orders.to_csv(args.output_dir / ORDERS_FILENAME, index=False)
    LOGGER.info("Saved generated datasets to %s", args.output_dir)


if __name__ == "__main__":
    main()
