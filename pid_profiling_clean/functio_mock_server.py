from __future__ import annotations

import json
import logging
import os
import shlex
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import subprocess

from unified_analysis import transform_path, unified_strategy
from function_lookup import function_lookup
from logging_setup import configure_program_logging

# Make repo-root modules importable when this file is executed directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from execute_query import start_server, stop_server, wait_for_server_ready  # noqa: E402


APP_EXECUTABLES = {
    "clickhouse": {
        "clickhouse-exe": "/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse",
        "clickhouse-server": [
            "/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse",
            "server",
        ],
        "clickhouse-client": [
            "/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse",
            "client",
            "--query",
        ],
    }
}


@dataclass(frozen=True)
class ProjectConfig:
    executable: str
    server_cmd: list[str]
    client_cmd: list[str]
    exclude_dirs: list[str]
    init_prefix: str
    custom_prefix: str


projectConfig = ProjectConfig(
    executable=APP_EXECUTABLES["clickhouse"]["clickhouse-exe"],
    server_cmd=APP_EXECUTABLES["clickhouse"]["clickhouse-server"],
    client_cmd=APP_EXECUTABLES["clickhouse"]["clickhouse-client"],
    exclude_dirs=[
        "/home/ubuntu/ClickHouse_debug/contrib",
        "/home/ubuntu/ClickHouse_debug/base/glibc-compatibility",
    ],
    init_prefix="/home/ubuntu/ClickHouse_debug",
    custom_prefix="/home/rocky/functio/ClickHouse",
)

ResultPayload = dict[str, object]


@dataclass
class QueryResult:
    success: bool
    error: Optional[str] = None


def execute_query(query: str) -> QueryResult:
    query_execution_cmd = projectConfig.client_cmd + [query]
    try:
        query_execution_result = subprocess.run(
            query_execution_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,  # 10 min timeout
        )
        if query_execution_result.returncode != 0:
            return QueryResult(success=False, error=query_execution_result.stderr.strip())
        return QueryResult(success=True)
    except subprocess.CalledProcessError as e:
        return QueryResult(success=False, error=str(getattr(e, "stderr", e)))


_analysis_lock = threading.Lock()


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _cmd_to_shell(cmd: list[str]) -> str:
    # unified_strategy ultimately runs `bash -c "$EXEC_CMD"`, so pass a shell string.
    return " ".join(shlex.quote(x) for x in cmd)


def _run_pipeline(queries: list[str], logger: logging.Logger) -> ResultPayload:
    if not queries:
        logger.warning("No queries provided - aborting")
        return {"result": ["No queries provided. Input query to continue"], "status": "fail"}

    logger.info("Starting ClickHouse server...")
    start_server(projectConfig.server_cmd[0])
    if not wait_for_server_ready(client_path=projectConfig.client_cmd[0], timeout=30):
        logger.error("Server not ready - aborting")
        stop_server()
        return {"result": ["Server not ready. Abort"], "status": "fail"}
    logger.info("ClickHouse server is ready")

    try:
        setup_queries = queries[:-1]
        logger.info(f"Executing {len(setup_queries)} setup queries...")
        for i, q in enumerate(setup_queries):
            logger.info(f"Executing setup query {i+1}/{len(setup_queries)}: {q[:100].strip()}...")
            res = execute_query(q)
            if not res.success:
                logger.error(f"Setup query {i+1} failed: {res.error}")
                return {"result": [f"One of the queries failed to execute: {res.error}"], "status": "fail"}

        last_query = queries[-1]
        logger.info(f"Main query to analyze: {last_query[:200].strip()}...")

        cmd_str = _cmd_to_shell(projectConfig.client_cmd + [last_query])
        logger.info(f"Profiling command: {cmd_str}")

        profiling_dir = _REPO_ROOT / "pid_profiling_clean"
        try:
            with _analysis_lock, _pushd(profiling_dir):
                hot_functions = unified_strategy(
                    cmd_str,
                    "report",
                    False,
                    True,
                    projectConfig.exclude_dirs,
                )
            logger.info(f"Flamegraph analysis complete. Found {len(hot_functions)} hot functions")
        except Exception as e:
            logger.error(f"Flamegraph analysis failed: {e}", exc_info=True)
            return {"result": [f"error running flamegraph analysis. Error message: {e}"], "status": "fail"}

        function_lookup_results: list[tuple[str, str]] = []
        logger.info("Looking up function definitions...")
        for function_name, file_path in hot_functions:
            logger.info(f"Looking up function: {function_name} in {file_path}")
            try:
                resolved_path = function_lookup(function_name, file_path, projectConfig.executable)
                if resolved_path is not None:
                    file_location = transform_path(
                        resolved_path,
                        projectConfig.init_prefix,
                        projectConfig.custom_prefix,
                    )
                    function_lookup_results.append((function_name, file_location))
                    logger.info(f"Found function: {function_name} -> {file_location}")
                else:
                    logger.warning(f"Function not found: {function_name}")
            except Exception as e:
                logger.error(f"Error looking up {function_name}: {e}", exc_info=True)

        if function_lookup_results:
            return {"result": function_lookup_results, "status": "success"}
        return {"result": function_lookup_results, "status": "fail"}

    finally:
        logger.info("Stopping ClickHouse server...")
        stop_server()
        logger.info("ClickHouse server stopped")


# DEFAULT_QUERIES: list[str] = [
#     """
# CREATE TABLE IF NOT EXISTS stress_test_5(
#     id UInt64,
#     random_data String,
#     val Float64
# ) ENGINE = MergeTree()
# ORDER BY id;
# """,
#     """
# INSERT INTO stress_test_5
# SELECT
#     number AS id,
#     hex(MD5(toString(number))) AS random_data,
#     rand() / 4294967295 AS val
# FROM numbers(100000);
# """,
#     """
# SELECT
#     id,
#     sin(val) * tan(val) + cos(id % 360) AS math_crunch,
#     SHA256(random_data || toString(math_crunch)) AS heavy_hash
# FROM stress_test_5
# WHERE
#     toString(id) LIKE '%5%'
# ORDER BY
#     math_crunch DESC
# LIMIT 1;
# """,
# ]

DEFAULT_QUERIES: list[str] = [
    "SELECT histogram(128)(randCanonical()) FROM numbers(100000000) FORMAT Null"
]

QUERIES_SQL_FILE = _REPO_ROOT / "pid_profiling_clean" / "queries_26th.sql"


def load_queries_from_file(sql_file: Path) -> list[str]:
    """Read queries from a SQL file, one query per line. Skips empty lines."""
    queries = []
    with open(sql_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("--"):  # skip empty lines and comments
                queries.append(line)
    return queries


if __name__ == "__main__":
    logging_handle = None
    try:
        run_timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        base_run_dir = _REPO_ROOT / "pid_profiling_clean" / "server_logs" / f"logs_{run_timestamp}"
        base_run_dir.mkdir(parents=True, exist_ok=True)

        default_log_path = base_run_dir / "full_run_log.log"
        logging_handle = configure_program_logging(
            enabled=True,
            log_file=str(default_log_path),
            level=logging.INFO,
            to_console=True,
            append=True,
            capture_prints=True,
            fmt="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info(f"Mock run started at {run_timestamp}")
        logger.info("=" * 60)

        # Load queries from SQL file
        queries = load_queries_from_file(QUERIES_SQL_FILE)
        logger.info(f"Loaded {len(queries)} queries from {QUERIES_SQL_FILE}")

        # Run each query and collect results
        all_results: dict[str, ResultPayload] = {}
        for i, query in enumerate(queries, start=1):
            logger.info("=" * 60)
            logger.info(f"Processing query {i}/{len(queries)}: {query[:80]}...")
            logger.info("=" * 60)

            # Each query is run as a single-query pipeline (no setup queries)
            payload = _run_pipeline([query], logger)
            all_results[query] = payload

            logger.info(f"Query {i} status: {payload.get('status')}")
            logger.info(f"Query {i} result count: {len(payload.get('result', []))}")

        # Save all results to JSON file
        results_file = base_run_dir / "profiling_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

        # Summary
        success_count = sum(1 for r in all_results.values() if r.get("status") == "success")
        logger.info("=" * 60)
        logger.info(f"Completed: {success_count}/{len(queries)} queries succeeded")
        logger.info("=" * 60)

        print(f"Results saved to: {results_file}")

    finally:
        if logging_handle and hasattr(logging_handle, "stop"):
            logging_handle.stop()
