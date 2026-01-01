from typing_extensions import ParamSpecArgs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unified_analysis import unified_strategy, transform_path
from function_lookup import function_lookup
import logging
from dataclasses import dataclass
from typing import Optional
import subprocess
from logging_setup import configure_program_logging
from datetime import datetime
from pathlib import Path
import server_manage



APP_EXECUTABLES = {"clickhouse":
{
    "clickhouse-exe": ["/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse"],
    "clickhouse-server": ["/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse", "server"],
    "clickhouse-client": ["/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse", "client", "--query"],

}
}
#TODO CHANGE THIS TO THE DATA PASSED THROUGH A SERVER
projectConfig = ProjectConfig(APP_EXECUTABLES["clickhouse"]["clickhouse-exe"],APP_EXECUTABLES["clickhouse"]["clickhouse-server"], APP_EXECUTABLES["clickhouse"]["clickhouse-client"], [["/home/ubuntu/ClickHouse_debug/contrib", "/home/ubuntu/ClickHouse_debug/base/glibc-compatibility/"]], "/home/ubuntu/ClickHouse_debug", "/home/rocky/functio/ClickHouse",)
#TODO check if passing the whole command as one string will work actually
db_instance = DBLifecycleManager(projectConfig.server_cmd, "localhost", "9000")
       

class ProjectConfig:
    def init(self, executable, server_cmd, client_cmd, exclude_dirs, init_prefix, custom_prefix):
        self.executable = executable
        self.server_cmd = server_cmd
        self.client_cmd = client_cmd
        self.exclude_dirs = exclude_dirs
        self.init_prefix = init_prefix
        self.custom_prefix = custom_prefix

# 1. Create a Pydantic model for request data validation
# This ensures the incoming JSON has a 'value' field that is a list of strings.
class RequestData(BaseModel):
    value: list[str]  # Changed from str to list[str]

# 2. Create a Pydantic model for the response
class ResponseData(BaseModel):
    result: list
    status: str

@dataclass
class QueryResult:
    success: bool
    error: Optional[str] = None


def execute_query(query:str):
    query_execution_cmd = projectConfig.client_cmd+[query]
    
    try:
        query_execution_result = subprocess.run(
            query_execution_cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=600 #10 min timeout
        )
        if query_execution_result.returncode != 0:
            return QueryResult(success=False, error=query_execution_result.stderr.strip())
        return QueryResult(success=True, error=None)
    except subprocess.CalledProcessError as e:
        return QueryResult(success=False, error=e.stderr)
# 3. Initialize the FastAPI app
app = FastAPI()

# 4. Define the API endpoint
@app.post("/process", response_model=ResponseData)
def process_data(data: RequestData):
    logging_handle = None
    try:
        # Setup logging for this request
        run_timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        base_run_dir = Path(f"server_logs/logs_{run_timestamp}")
        base_run_dir.mkdir(parents=True, exist_ok=True)
        
        default_log_path = base_run_dir / "full_run_log.log"
        logging_handle = configure_program_logging(
            enabled=True,
            log_file=str(default_log_path),
            level=logging.INFO,
            to_console=False,
            append=True,
            capture_prints=True,
            fmt='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info(f"New request received at {run_timestamp}")
        logger.info("=" * 60)
        
        # Start ClickHouse server
        logger.info("Starting ClickHouse server...")
        db_instance.ensure_up()
        logger.info("ClickHouse server started successfully")
        
        data_value = data.value
        logger.info(f"Received {len(data_value)} queries")
        
        function_lookup_results = []
        if len(data_value) == 0:
            logger.warning("No queries provided - aborting")
            return ResponseData(result=['No queries provided. Input query to continue'], status="fail")
        
        # Execute setup queries
        setup_queries = data_value[:-1]
        logger.info(f"Executing {len(setup_queries)} setup queries...")
        for i, query in enumerate(setup_queries):
            logger.info(f"Executing setup query {i+1}/{len(setup_queries)}: {query[:100]}...")
            queryResult = execute_query(query)
            if not queryResult.success:
                logger.error(f"Setup query {i+1} failed: {queryResult.error}")
                return ResponseData(result=[f'One of the queries failed to execute: {queryResult.error}'], status="fail")
            logger.info(f"Setup query {i+1} completed successfully")
        
        # Execute main query with flamegraph analysis
        last_query = data_value[-1]
        logger.info(f"Main query to analyze: {last_query}")
        last_cmd = projectConfig.client_cmd+[shlex.quote(last_query)]
        try:
            logger.info("Starting flamegraph analysis (unified_strategy)...")
            hot_functions = unified_strategy(
                last_cmd,
                "report",
                False,
                True,
                projectConfig.exclude_dirs
            )
            logger.info(f"Flamegraph analysis complete. Found {len(hot_functions)} hot functions")
        except Exception as e:
            logger.error(f"Flamegraph analysis failed: {e}", exc_info=True)
            return ResponseData(result=[f"error running flamegraph analysis. Error message: {e}"], status="fail")
        
        # Look up each hot function
        logger.info("Looking up function definitions...")
        for function_name, file_path in hot_functions:
            logger.info(f"Looking up function: {function_name} in {file_path}")
            try:
                #returns file path on the perf cloud
                result = function_lookup(function_name, file_path, projectConfig.executable)
                if result is not None:
                    file_location = transform_path(
                        result,
                        projectConfig.init_prefix,
                        projectConfig.custom_prefix,
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
        db_instance.shutdown()
        logger.info("ClickHouse server stopped")
        
        if function_lookup_results:
            logger.info("Request completed successfully")
            return ResponseData(result=function_lookup_results, status="success")
        else:
            logger.warning("No function lookup results - returning fail status")
            return ResponseData(result=function_lookup_results, status="fail")

    except Exception as e:
        try:
            logging.getLogger(__name__).error(f"Unexpected error: {e}", exc_info=True)
        except:
            pass
        return ResponseData(result=[str(e)], status="fail")
    
    finally:
        # Clean up logging
        if logging_handle and hasattr(logging_handle, 'stop'):
            logging_handle.stop()

        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
    #process_data(None)
