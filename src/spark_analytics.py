"""Analyze synthetic e-commerce data with PySpark."""

import argparse
import logging
import os
from pathlib import Path
import sys

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

from src.config import (
    CUSTOMERS_FILENAME,
    ORDERS_FILENAME,
    PRODUCTS_FILENAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    configure_logging,
)

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Provide Spark session management and e-commerce sales analyses."""

    def __init__(self) -> None:
        """Create an analytics helper without starting a Spark session."""
        self.spark: SparkSession | None = None

    def create_spark_session(self) -> SparkSession:
        """Create a local Spark session with 4 GB memory, AQE, and Kryo."""
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
        self.spark = (
            SparkSession.builder.appName("EcommerceSalesAnalytics")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.sql.adaptive.enabled", "true")
            .config(
                "spark.serializer",
                "org.apache.spark.serializer.KryoSerializer",
            )
            .getOrCreate()
        )
        LOGGER.info("Created Spark session for local sales analytics")
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet dataset from ``path`` into a Spark DataFrame."""
        if self.spark is None:
            raise RuntimeError(
                "Create a Spark session before loading Parquet data."
            )
        LOGGER.info("Loading Parquet data from %s", path)
        return self.spark.read.parquet(str(path))

    def _orders_with_product_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Join order lines to product keys and calculate line revenue."""
        if "unit_price" not in orders_df.columns and "price" not in products_df.columns:
            raise ValueError(
                "Orders must include unit_price or products must include price."
            )
        orders = orders_df.alias("orders")
        product_columns = ["product_id"]
        if "price" in products_df.columns:
            product_columns.append(F.col("price").alias("_product_price"))
        products = products_df.select(*product_columns).alias("products")
        unit_price = (
            F.col("orders.unit_price")
            if "unit_price" in orders_df.columns
            else F.col("products._product_price")
        )
        return (
            orders.join(
                products,
                F.col("orders.product_id") == F.col("products.product_id"),
                "inner",
            )
            .select(
                F.col("orders.customer_id").alias("customer_id"),
                F.col("orders.product_id").alias("product_id"),
                F.col("orders.order_date").alias("order_date"),
                F.col("orders.quantity").cast("long").alias("quantity"),
                (
                    F.col("orders.quantity").cast(DecimalType(18, 2))
                    * unit_price.cast(DecimalType(18, 2))
                ).alias("revenue"),
            )
        )

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top ``n`` customers ordered by total order revenue."""
        if n < 0:
            raise ValueError("n must be zero or greater")
        revenue_orders = self._orders_with_product_revenue(orders_df, products_df)
        return (
            revenue_orders.groupBy("customer_id")
            .agg(F.sum("revenue").alias("total_spend"))
            .orderBy(F.desc("total_spend"), F.asc("customer_id"))
            .limit(n)
        )

    def sales_by_category(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Return total revenue and units sold grouped by product category."""
        if "unit_price" not in orders_df.columns and "price" not in products_df.columns:
            raise ValueError(
                "Orders must include unit_price or products must include price."
            )
        orders = orders_df.alias("orders")
        product_columns = ["product_id", "category"]
        if "price" in products_df.columns:
            product_columns.append(F.col("price").alias("_product_price"))
        products = products_df.select(*product_columns).alias("products")
        joined = orders.join(
            products,
            F.col("orders.product_id") == F.col("products.product_id"),
            "inner",
        )
        unit_price = (
            F.col("orders.unit_price")
            if "unit_price" in orders_df.columns
            else F.col("products._product_price")
        )
        revenue = (
            F.col("orders.quantity").cast(DecimalType(18, 2))
            * unit_price.cast(DecimalType(18, 2))
        )
        return (
            joined.groupBy(F.col("products.category").alias("category"))
            .agg(
                F.sum(revenue).alias("total_revenue"),
                F.sum(F.col("orders.quantity").cast("long")).alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"), F.asc("category"))
        )

    def monthly_trends(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Return monthly revenue and month-over-month revenue growth percent."""
        revenue_orders = self._orders_with_product_revenue(orders_df, products_df)
        monthly_revenue = (
            revenue_orders.withColumn(
                "month",
                F.date_format(F.to_date("order_date"), "yyyy-MM"),
            )
            .groupBy("month")
            .agg(F.sum("revenue").alias("revenue"))
        )
        month_window = Window.orderBy("month")
        with_previous_month = monthly_revenue.withColumn(
            "previous_month_revenue",
            F.lag("revenue").over(month_window),
        )
        previous_revenue = F.col("previous_month_revenue")
        growth = F.when(
            previous_revenue.isNotNull() & (previous_revenue != 0),
            (
                (F.col("revenue") - previous_revenue)
                / previous_revenue
                * F.lit(100)
            ).cast("double"),
        )
        return (
            with_previous_month.withColumn("revenue_growth_pct", growth)
            .orderBy("month")
        )


def _read_csv(spark: SparkSession, file_path: Path) -> DataFrame:
    """Read a headered CSV file and fail clearly if it is missing."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Required input CSV was not found: {file_path}")
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(file_path))
    )


def _write_csv(dataframe: DataFrame, output_path: Path) -> None:
    """Write a Spark DataFrame as a single headered CSV part file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(output_path))
    )
    LOGGER.info("Wrote analytics output to %s", output_path)


def run_analytics(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """Build and save revenue, product, and customer summary datasets."""
    analytics = SalesAnalytics()
    spark = analytics.create_spark_session()
    try:
        customers = _read_csv(spark, input_dir / CUSTOMERS_FILENAME)
        products = _read_csv(spark, input_dir / PRODUCTS_FILENAME)
        if "product_name" not in products.columns and "name" in products.columns:
            products = products.withColumnRenamed("name", "product_name")
        orders = _read_csv(spark, input_dir / ORDERS_FILENAME)
        if "unit_price" not in orders.columns:
            if "price" not in products.columns:
                raise ValueError(
                    "Orders must include unit_price or products must include price."
                )
            orders = orders.join(
                products.select("product_id", F.col("price").alias("_product_price")),
                on="product_id",
                how="left",
            ).withColumnRenamed("_product_price", "unit_price")
        orders = (
            orders.withColumn("order_date", F.to_date("order_date"))
            .withColumn("quantity", F.col("quantity").cast("integer"))
            .withColumn("unit_price", F.col("unit_price").cast(DecimalType(10, 2)))
        )

        sales_by_category = analytics.sales_by_category(orders, products)
        sales_by_month = analytics.monthly_trends(orders, products).withColumnRenamed(
            "revenue", "total_revenue"
        )
        top_products = (
            orders.join(products, on="product_id", how="inner")
            .withColumn(
                "revenue",
                F.col("quantity") * F.col("unit_price"),
            )
            .groupBy("product_id", "product_name", "category")
            .agg(
                F.sum("revenue").alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )
        customer_totals = (
            orders.withColumn("revenue", F.col("quantity") * F.col("unit_price"))
            .groupBy("customer_id")
            .agg(
                F.countDistinct("order_id").alias("order_count"),
                F.sum("revenue").alias("total_spend"),
            )
        )
        customer_summary = (
            customers.join(customer_totals, on="customer_id", how="left")
            .fillna({"order_count": 0, "total_spend": 0})
            .orderBy("customer_id")
        )

        outputs = {
            "sales_by_category": output_dir / "sales_by_category",
            "sales_by_month": output_dir / "sales_by_month",
            "top_products": output_dir / "top_products",
            "customer_summary": output_dir / "customer_summary",
        }
        _write_csv(sales_by_category, outputs["sales_by_category"])
        _write_csv(sales_by_month, outputs["sales_by_month"])
        _write_csv(top_products, outputs["top_products"])
        _write_csv(customer_summary, outputs["customer_summary"])
        return outputs
    finally:
        spark.stop()
        analytics.spark = None


def _parse_args() -> argparse.Namespace:
    """Parse command-line input and output directory options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing generated CSV input files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for Spark analytics output",
    )
    return parser.parse_args()


def main() -> None:
    """Run the PySpark analytics pipeline from the command line."""
    configure_logging()
    args = _parse_args()
    run_analytics(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
