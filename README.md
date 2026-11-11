# E-commerce Data Pipeline

Generate synthetic e-commerce customers, products, and orders, then use PySpark
to produce datasets for business analysis. The data is fictional and intended
for development, demos, and testing only.

## Project structure

```text
.
├── data/
│   ├── processed/                 # PySpark analysis outputs
│   └── raw/                       # Generated CSV input data
├── notebooks/                     # Space for Jupyter notebooks
├── src/
│   ├── _init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Use Python 3.10 or newer and install Java 17 or newer for PySpark. On Windows,
configure `HADOOP_HOME` to a Hadoop installation that includes `bin\winutils.exe`
so Spark can write local output files.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the pipeline

Generate the default 100,000 customers, 10,000 products, and 1,000,000 orders:

```powershell
python main.py
```

The generator displays progress bars and uses more memory at the default sizes.
The script saves `customers.parquet`, `products.parquet`, and `orders.parquet`
under `data/raw/` and reports generation time and each file's size. For a
smaller deterministic run, pass for example:

```powershell
python main.py --customers 500 --products 50 --orders 2000 --seed 7
```

Analyze the generated CSV files with Spark:

```powershell
python -m src.spark_analytics
```

Run the interactive Parquet analytics and display each result with Spark:

```powershell
python run_analytics.py
```

This loads `customers.parquet`, `products.parquet`, and `orders.parquet` from
`data/raw/`, prints timings, and stops its Spark session when complete.

The analytics step writes four CSV output directories under
`data/processed/`: `sales_by_category`, `sales_by_month`, `top_products`, and
`customer_summary`. PySpark writes
CSV datasets as directories containing one or more part files.

To use custom input/output directories, pass `--input-dir` and `--output-dir`
to `spark_analytics`.

## Generate pandas DataFrames

`SyntheticDataGenerator` returns customers, products, and orders as pandas
DataFrames. The CLI writes those frames to `data/raw/`. The order generator
assigns 80% of orders to the first 20% of customers and uses Pareto weighting
within each group.

```python
from src.config import RAW_DATA_DIR
from src.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=7)
customers, products, orders = generator.generate_all()

customers.to_csv(RAW_DATA_DIR / "customers.csv", index=False)
products.to_csv(RAW_DATA_DIR / "products.csv", index=False)
orders.to_csv(RAW_DATA_DIR / "orders.csv", index=False)
```

The DataFrame generator is memory-intensive at its default sizes; lower the
counts passed to `generate_all()` if needed.

The `SalesAnalytics` class in `src/spark_analytics.py` also exposes reusable
DataFrame methods for top customers, category sales, and monthly revenue
growth. Call `create_spark_session()` before `load_parquet(path)`. The
analytics methods expect order columns `customer_id`, `product_id`, `quantity`,
and `order_date`, plus `unit_price` on orders or `price` on products; product
columns include `product_id` and `category`.

## Benchmark file formats

Compare CSV, XLSX, Parquet (fastparquet), ORC (PyArrow), and Feather on a
500,000-row sample:

```powershell
python benchmark_file_formats.py
```

Use `--rows` to run a smaller test and `--output-dir` to choose where benchmark
files are written. Metrics include file size, write/read time, Python-traced
peak memory, CPU time, an estimated energy value at a 65 W TDP, and file-size
savings relative to CSV. Tracemalloc excludes native-library allocations; the
energy value is an estimate based on process CPU time, not measured power.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The raw and processed data folders are kept in the repository, while generated
data files are ignored by Git.
