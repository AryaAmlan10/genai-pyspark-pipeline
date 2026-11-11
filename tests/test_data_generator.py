"""Tests for synthetic e-commerce data generation."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data_generator import SyntheticDataGenerator, generate_dataset


class GenerateDatasetTests(unittest.TestCase):
    """Validate generated files and their cross-file relationships."""

    def test_generates_reproducible_related_csv_files(self) -> None:
        """Write requested row counts with customer/product references intact."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            paths = generate_dataset(
                output_dir=output_dir / "first",
                customer_count=3,
                product_count=2,
                order_count=5,
                seed=17,
            )
            repeated_paths = generate_dataset(
                output_dir=output_dir / "repeated",
                customer_count=3,
                product_count=2,
                order_count=5,
                seed=17,
            )

            with paths["customers"].open(encoding="utf-8", newline="") as file:
                customers = list(csv.DictReader(file))
            with paths["products"].open(encoding="utf-8", newline="") as file:
                products = list(csv.DictReader(file))
            with paths["orders"].open(encoding="utf-8", newline="") as file:
                orders = list(csv.DictReader(file))

            self.assertEqual(len(customers), 3)
            self.assertEqual(len(products), 2)
            self.assertEqual(len(orders), 5)
            self.assertTrue(
                {order["customer_id"] for order in orders}
                <= {customer["customer_id"] for customer in customers}
            )
            self.assertTrue(
                {order["product_id"] for order in orders}
                <= {product["product_id"] for product in products}
            )
            self.assertEqual(set(paths), {"customers", "products", "orders"})
            for name, file_path in paths.items():
                self.assertEqual(file_path.read_bytes(), repeated_paths[name].read_bytes())

    def test_rejects_empty_entity_datasets(self) -> None:
        """Reject zero-sized inputs that cannot produce valid order references."""
        with self.assertRaises(ValueError):
            generate_dataset(customer_count=0)


class SyntheticDataGeneratorTests(unittest.TestCase):
    """Validate generated DataFrame schemas, ranges, and order allocation."""

    def test_generates_related_dataframes_with_an_80_20_order_split(self) -> None:
        """Generate small deterministic data and verify requested constraints."""
        generator = SyntheticDataGenerator(seed=23)
        customers = generator.generate_customers(20)
        products = generator.generate_products(10)
        orders = generator.generate_orders(customers, products, 1_000)

        self.assertEqual(
            list(customers.columns),
            [
                "customer_id",
                "name",
                "email",
                "age",
                "city",
                "country",
                "registration_date",
            ],
        )
        self.assertTrue(customers["age"].between(18, 90).all())
        self.assertTrue(customers["email"].str.contains("@").all())
        self.assertTrue(products["price"].between(10, 500).all())
        self.assertTrue(products["stock"].between(0, 1_000).all())
        self.assertTrue(products["rating"].between(1, 5).all())
        self.assertTrue(orders["quantity"].between(1, 10).all())
        self.assertTrue(set(orders["customer_id"]) <= set(customers["customer_id"]))
        self.assertTrue(set(orders["product_id"]) <= set(products["product_id"]))
        high_volume_customer_ids = set(customers["customer_id"].iloc[:4])
        high_volume_orders = orders["customer_id"].isin(high_volume_customer_ids).sum()
        self.assertEqual(high_volume_orders, 800)

    def test_rejects_empty_inputs_for_order_generation(self) -> None:
        """Require at least one customer and product before creating orders."""
        generator = SyntheticDataGenerator(seed=1)
        customers = generator.generate_customers(1)
        products = generator.generate_products(1)
        with self.assertRaises(ValueError):
            generator.generate_orders(customers.iloc[0:0], products, count=1)


if __name__ == "__main__":
    unittest.main()
