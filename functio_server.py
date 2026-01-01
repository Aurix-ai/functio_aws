from typing_extensions import ParamSpecArgs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fl_analyze_unified import unified_strategy, transform_path
from execute_query import *
from function_lookup import function_lookup
import logging
from dataclasses import dataclass
from typing import Optional
import subprocess

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
            return QueryResult(success=False, error=query_execution_result.stderr.strip())
        return QueryResult(success=True, error=None)
    except subprocess.CalledProcessError as e:
        return QueryResult(success=False, error=e.stderr)
# 3. Initialize the FastAPI app
app = FastAPI()

# 4. Define the API endpoint
@app.post("/process", response_model=ResponseData)
def process_data(data: RequestData):
    try:
        start_server()
        data_value = data.value
#         data_value=(
#             """
#             CREATE TABLE IF NOT EXISTS stress_test_5(
#     id UInt64,
#     random_data String,
#     val Float64
# ) ENGINE = MergeTree()
# ORDER BY id;
#             """,
#             """
#             INSERT INTO stress_test_5
# SELECT 
#     number AS id,
#     hex(MD5(toString(number))) AS random_data,
#     rand() / 4294967295 AS val                 
# FROM numbers(100000);
#             """,
            
#             """
#             SELECT 
#     id,
#     sin(val) * tan(val) + cos(id % 360) AS math_crunch,
#     SHA256(random_data || toString(math_crunch)) AS heavy_hash
# FROM stress_test_5
# WHERE 
#     toString(id) LIKE '%5%' 
# ORDER BY 
#     math_crunch DESC
# LIMIT 1;
#             """)
        function_lookup_results = []
        if len(data_value)==0:
            print("data_value len can't be 0")
            return ResponseData(result=['No queries provided. Input query to continue'], status="fail")
        if not wait_for_server_ready():
            print("Server not ready. Abort")
            return ResponseData(result=['Server not ready. Abort'], status="fail")
        setup_queries = data_value[:-1]
        for query in setup_queries:
            queryResult = execute_query(query)
            if not queryResult.success:
                print(f'One of the queries failed to execute: {queryResult.error}')
                return ResponseData(result=[f'One of the queries failed to execute: {queryResult.error}'], status="fail")
        last_query = data_value[-1]
        print(last_query)
        try:
            hot_functions = unified_strategy(
                last_query,
                "report",
                False,
                True,
                ["/home/ubuntu/ClickHouse_debug/contrib", "/home/ubuntu/ClickHouse_debug/base/glibc-compatibility/"],
                prefix_path="/home/ubuntu/ClickHouse_debug",
                custom_prefix="/home/ubuntu/ClickHouse_debug",
            )
        except Exception as e:
            return ResponseData(result=f"error running flamegraph analysis. Error message: {e}", status="fail")
        for function_name, file_path in hot_functions:
            print(f"\n--- Looking up function: {function_name} in {file_path} ---")
            try:
                result = function_lookup(function_name, file_path)
                if result is not None:
                    file_location = transform_path(
                        result[0],
                        "/home/ubuntu/ClickHouse_debug",
                        "/home/rocky/functio/ClickHouse",
                    )
                    function_lookup_results.append((function_name, file_location))
            except Exception as e:
                print(f"Error looking up {function_name}: {e}")

        print(f"\nFunction lookup results: {function_lookup_results}")

        stop_server()
        if function_lookup_results:
            return ResponseData(result=function_lookup_results, status="success")
        else:
            return ResponseData(result=function_lookup_results, status="fail")

    except Exception as e:
        return ResponseData(result=[str(e)], status="fail")

        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
    #process_data(None)
