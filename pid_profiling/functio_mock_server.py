from typing_extensions import ParamSpecArgs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unified_analysis import unified_strategy, transform_path
from execute_query import *
from function_lookup import function_lookup
import logging
from dataclasses import dataclass
from typing import Optional
import subprocess
from logging_setup import configure_program_logging
from datetime import datetime
from pathlib import Path


CPP_APP_EXECUTABLE = '/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse'

def execute_query(query:str):
    query_execution_cmd = [
        CPP_APP_EXECUTABLE,
        "client",
        "--query", query
    ]
    try:
        query_execution_result = subprocess.run(
            query_execution_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=600 #10 min timeout
        )
        if query_execution_result.returncode != 0:
            print(query_execution_result.stderr.strip())
            return False
        print(f"executed correctly query {query}")
        return True
    except subprocess.CalledProcessError as e:
        print("failed")
        return False
# 3. Initialize the FastAPI app
app = FastAPI()

# 4. Define the API endpoint
def process_data():
    logging_handle = None
    try:
        # Setup logging for this run
        run_timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        base_run_dir = Path(f"server_logs/logs_{run_timestamp}")
        base_run_dir.mkdir(parents=True, exist_ok=True)
        
        default_log_path = base_run_dir / "full_run_log.log"
        logging_handle = configure_program_logging(
            enabled=True,
            log_file=str(default_log_path),
            level=logging.INFO,
            to_console=True,  # Also show in console for mock server
            append=True,
            capture_prints=True,
            fmt='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info(f"Mock server run started at {run_timestamp}")
        logger.info("=" * 60)
        
        # Start ClickHouse server
        logger.info("Starting ClickHouse server...")
        start_server()
        logger.info("ClickHouse server started successfully")
        
        data_value=(
            """
            CREATE TABLE IF NOT EXISTS stress_test_5(
    id UInt64,
    random_data String,
    val Float64
) ENGINE = MergeTree()
ORDER BY id;
            """,
            """
            INSERT INTO stress_test_5
SELECT 
    number AS id,
    hex(MD5(toString(number))) AS random_data,
    rand() / 4294967295 AS val                 
FROM numbers(100000);
            """,
            
            """
            SELECT 
    id,
    sin(val) * tan(val) + cos(id % 360) AS math_crunch,
    SHA256(random_data || toString(math_crunch)) AS heavy_hash
FROM stress_test_5
WHERE 
    toString(id) LIKE '%5%' 
ORDER BY 
    math_crunch DESC
LIMIT 1;
            """)
        
        logger.info(f"Processing {len(data_value)} queries")
        function_lookup_results = []
        
        if len(data_value) == 0:
            logger.warning("No queries provided - aborting")
            return
        
        # Wait for server to be ready
        logger.info("Waiting for server to be ready...")
        if not wait_for_server_ready():
            logger.error("Server not ready - aborting")
            return
        logger.info("Server is ready")
        
        # Execute setup queries
        setup_queries = data_value[:-1]
        logger.info(f"Executing {len(setup_queries)} setup queries...")
        for i, query in enumerate(setup_queries):
            logger.info(f"Executing setup query {i+1}/{len(setup_queries)}: {query[:80].strip()}...")
            queryResult = execute_query(query)
            if not queryResult:
                logger.error(f"Setup query {i+1} failed")
                return
            logger.info(f"Setup query {i+1} completed successfully")
        
        # Execute main query with flamegraph analysis
        last_query = data_value[-1]
        logger.info(f"Main query to analyze: {last_query[:100].strip()}...")
        
        try:
            logger.info("Starting flamegraph analysis (unified_strategy)...")
            hot_functions = unified_strategy(
                last_query,
                "report",
                False,
                True,
                ["/home/ubuntu/ClickHouse_debug/contrib", "/home/ubuntu/ClickHouse_debug/base/glibc-compatibility/"],
                prefix_path="/home/ubuntu/ClickHouse_debug",
                custom_prefix="/home/ubuntu/ClickHouse_debug",
            )
            logger.info(f"Flamegraph analysis complete. Found {len(hot_functions)} hot functions")
        except Exception as e:
            logger.error(f"Flamegraph analysis failed: {e}", exc_info=True)
            return
        
        # Look up each hot function
        logger.info("Looking up function definitions...")
        for function_name, file_path in hot_functions:
            logger.info(f"Looking up function: {function_name} in {file_path}")
            try:
                result = function_lookup(function_name, file_path)
                if result is not None:
                    file_location = transform_path(
                        result[0],
                        "/home/ubuntu/ClickHouse_debug",
                        "/home/rocky/functio/ClickHouse",
                    )
                    function_lookup_results.append((function_name, file_location))
                    logger.info(f"Found function: {function_name} -> {file_location}")
                else:
                    logger.warning(f"Function not found: {function_name}")
            except Exception as e:
                logger.error(f"Error looking up {function_name}: {e}")

        logger.info(f"Function lookup complete. Found {len(function_lookup_results)} functions")
        logger.info(f"Results: {function_lookup_results}")

        # Stop server
        logger.info("Stopping ClickHouse server...")
        stop_server()
        logger.info("ClickHouse server stopped")
        
        if function_lookup_results:
            logger.info("Run completed successfully")
            logger.info(f"Final results: {function_lookup_results}")
        else:
            logger.warning("No function lookup results - run failed")

    except Exception as e:
        try:
            logging.getLogger(__name__).error(f"Unexpected error: {e}", exc_info=True)
        except:
            pass
        print(f"general fail: {e}")
    
    finally:
        # Clean up logging
        if logging_handle and hasattr(logging_handle, 'stop'):
            logging_handle.stop()

        
if __name__ == "__main__":
    process_data()
