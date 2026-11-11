"""Benchmark common tabular file formats for size, time, memory, and CPU."""

import argparse
import logging
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.orc as orc
import tzdata

LOGGER = logging.getLogger(__name__)
DEFAULT_ROW_COUNT = 500_000
CPU_TDP_WATTS = 65.0
BYTES_PER_MB = 1_000_000
EXCEL_MAX_ROWS = 1_048_576
CATEGORIES = np.array(["Electronics", "Clothing", "Home", "Sports", "Books"])


def _write_orc(dataframe: pd.DataFrame, file_path: Path) -> None:
    """Write a pandas DataFrame to ORC, exposing Python tzdata to PyArrow."""
    timezone_data_dir = Path(tzdata.__file__).parent / "zoneinfo"
    os.environ.setdefault("TZDIR", str(timezone_data_dir))
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    orc.write_table(table, str(file_path))


def _read_orc(file_path: Path) -> pd.DataFrame:
    """Read an ORC file into pandas using PyArrow."""
    timezone_data_dir = Path(tzdata.__file__).parent / "zoneinfo"
    os.environ.setdefault("TZDIR", str(timezone_data_dir))
    return orc.ORCFile(str(file_path)).read().to_pandas()


@dataclass(frozen=True)
class FormatBenchmark:
    """Store measured performance and size data for one file format."""

    format_name: str
    file_size_mb: float
    write_seconds: float
    read_seconds: float
    peak_memory_mb: float
    cpu_seconds: float
    energy_wh: float


def create_benchmark_dataframe(
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Create the requested synthetic benchmark DataFrame."""
    if row_count < 1:
        raise ValueError("row_count must be at least 1")
    if row_count + 1 > EXCEL_MAX_ROWS:
        raise ValueError(
            f"XLSX supports at most {EXCEL_MAX_ROWS - 1:,} data rows per sheet"
        )

    rng = np.random.default_rng(seed)
    dates = np.datetime64("2020-01-01", "D") + rng.integers(
        0, 365 * 6, size=row_count
    ).astype("timedelta64[D]")
    identifiers = np.arange(1, row_count + 1, dtype=np.int64)
    return pd.DataFrame(
        {
            "id": identifiers,
            "name": [f"Customer {identifier}" for identifier in identifiers],
            "email": [f"customer{identifier}@example.com" for identifier in identifiers],
            "amount": np.round(rng.uniform(5, 1_000, size=row_count), 2),
            "date": pd.to_datetime(dates).astype("datetime64[ns]"),
            "category": rng.choice(CATEGORIES, size=row_count),
        }
    )


def _measure_operation(operation: Callable[[], object]) -> tuple[float, float, float]:
    """Return wall time, process CPU time, and peak traced memory for an operation."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        operation()
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return wall_seconds, cpu_seconds, peak_bytes / BYTES_PER_MB


def benchmark_format(
    dataframe: pd.DataFrame,
    format_name: str,
    file_path: Path,
) -> FormatBenchmark:
    """Write and read a dataset in one format and collect benchmark metrics."""
    if format_name == "CSV":
        write = lambda: dataframe.to_csv(file_path, index=False)
        read = lambda: pd.read_csv(file_path, parse_dates=["date"])
    elif format_name == "XLSX":
        write = lambda: dataframe.to_excel(
            file_path,
            index=False,
            engine="openpyxl",
        )
        read = lambda: pd.read_excel(
            file_path,
            engine="openpyxl",
            parse_dates=["date"],
        )
    elif format_name == "Parquet (fastparquet)":
        write = lambda: dataframe.to_parquet(
            file_path,
            engine="fastparquet",
            index=False,
        )
        read = lambda: pd.read_parquet(file_path, engine="fastparquet")
    elif format_name == "ORC (PyArrow)":
        write = lambda: _write_orc(dataframe, file_path)
        read = lambda: _read_orc(file_path)
    elif format_name == "Feather":
        write = lambda: dataframe.to_feather(file_path)
        read = lambda: feather.read_feather(file_path)
    else:
        raise ValueError(f"Unsupported benchmark format: {format_name}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    write_seconds, write_cpu_seconds, write_peak_mb = _measure_operation(write)
    read_seconds, read_cpu_seconds, read_peak_mb = _measure_operation(read)
    file_size_mb = file_path.stat().st_size / BYTES_PER_MB
    cpu_seconds = write_cpu_seconds + read_cpu_seconds
    return FormatBenchmark(
        format_name=format_name,
        file_size_mb=file_size_mb,
        write_seconds=write_seconds,
        read_seconds=read_seconds,
        peak_memory_mb=max(write_peak_mb, read_peak_mb),
        cpu_seconds=cpu_seconds,
        energy_wh=cpu_seconds * CPU_TDP_WATTS / 3_600,
    )


def run_benchmarks(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> list[FormatBenchmark]:
    """Benchmark all requested file formats and return their measurements."""
    formats = (
        ("CSV", ".csv"),
        ("XLSX", ".xlsx"),
        ("Parquet (fastparquet)", ".parquet"),
        ("ORC (PyArrow)", ".orc"),
        ("Feather", ".feather"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[FormatBenchmark] = []
    for format_name, extension in formats:
        file_path = output_dir / f"benchmark{extension}"
        LOGGER.info("Benchmarking %s at %s", format_name, file_path)
        results.append(benchmark_format(dataframe, format_name, file_path))
    return results


def print_comparison_table(results: list[FormatBenchmark]) -> None:
    """Print measurements and file-size savings relative to the CSV baseline."""
    csv_result = next(
        (result for result in results if result.format_name == "CSV"),
        None,
    )
    if csv_result is None:
        raise ValueError("Results must include the CSV baseline")

    headers = (
        "Format",
        "Size (MB)",
        "Write (s)",
        "Read (s)",
        "Peak memory* (MB)",
        "CPU (s)",
        "Energy (Wh)",
        "Savings vs CSV",
    )
    rows = [
        (
            result.format_name,
            f"{result.file_size_mb:.2f}",
            f"{result.write_seconds:.2f}",
            f"{result.read_seconds:.2f}",
            f"{result.peak_memory_mb:.2f}",
            f"{result.cpu_seconds:.2f}",
            f"{result.energy_wh:.4f}",
            f"{(csv_result.file_size_mb - result.file_size_mb) / csv_result.file_size_mb * 100:.2f}%",
        )
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "-+-".join("-" * width for width in widths)
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print("* Peak traced Python memory (tracemalloc); excludes native allocations.")
    print(f"Energy estimate assumes a constant {CPU_TDP_WATTS:.0f} W CPU TDP.")


def _parse_args() -> argparse.Namespace:
    """Parse row count, output directory, and random seed arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "processed" / "format_benchmark",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Create a benchmark dataset, run format benchmarks, and print results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    try:
        generation_started = time.perf_counter()
        dataframe = create_benchmark_dataframe(args.rows, args.seed)
        LOGGER.info(
            "Generated %d benchmark rows in %.2f seconds",
            len(dataframe),
            time.perf_counter() - generation_started,
        )
        results = run_benchmarks(dataframe, args.output_dir)
        print_comparison_table(results)
    except Exception:
        LOGGER.exception("File-format benchmark failed")
        raise


if __name__ == "__main__":
    main()
