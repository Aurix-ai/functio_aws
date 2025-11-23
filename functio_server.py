from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fl_analyze_unified import unified_strategy
from execute_query import start_server, execute_query, stop_server
from function_lookup import function_lookup
import logging

# 1. Create a Pydantic model for request data validation
# This ensures the incoming JSON has a 'value' field that is a list of strings.
class RequestData(BaseModel):
    value: list[str]  # Changed from str to list[str]

# 2. Create a Pydantic model for the response
class ResponseData(BaseModel):
    result: list
    status: str

# 3. Initialize the FastAPI app
app = FastAPI()

# 4. Define the API endpoint
@app.post("/process", response_model=ResponseData)
def process_data(data: RequestData):
    """
    Receives a list of query strings, processes only the last one, and returns results.
    """
    try:
        #print(f"Received data: {data}")
        start_server()
        #mock data
        data_value = (
    "CREATE DATABASE IF NOT EXISTS mydb",
    "USE mydb",
    "CREATE TABLE multiples ( value UInt64 ) ENGINE = MergeTree ORDER BY value",
    "INSERT INTO multiples SELECT 7 * number AS value FROM system.numbers LIMIT 1000000",
    "WITH 1000000 AS N SELECT arrayJoin(topK(500)(intHash64(value) % 1000000)) FROM multiples WHERE value < N FORMAT Null;",
    "SELECT * FROM multiplesss"
)

    #change to data.value back later
        if(len(data_value)>1):
            for query in data_value[0:len(data_value)-1]:
                success, output, error = execute_query(query)
                print(f"Executed query: {query}")
                print(f"Success: {success}")
                print(f"Output: {output}")
                print(f"Error: {error}")
            # we need to run flamegraph strategy on the last query only
            last_query = data_value[-1]
            hot_queries = unified_strategy(last_query, "report", False, True, ["/home/ubuntu/ClickHouse_debug/contrib"], prefix_path="/home/ubuntu/ClickHouse_debug", custom_prefix="/home/ubuntu/ClickHouse_debug")
            # hot_queries is a list of tuples [(function_name, full_path), ...]
            
            # Call function_lookup for each (function, location) pair
            function_lookup_results = []
            for function_name, file_path in hot_queries:
                print(f"\n--- Looking up function: {function_name} in {file_path} ---")
                try:
                    result = function_lookup(function_name, file_path)
                    if result is not None:
                        function_lookup_results.append({
                            "function": function_name,
                            "file": result[0],
                            "lookup_result": result[1]
                        })
                except Exception as e:
                    print(f"Error looking up {function_name}: {e}")
                    function_lookup_results.append({
                        "function": function_name,
                        "file": file_path,
                        "error": str(e)
                    })
        
        print(f"\nFunction lookup results: {function_lookup_results}")
        logging.info('FINISHED')

        stop_server()
        if function_lookup_results is not None:
            return ResponseData(result=function_lookup_results, status="success")
        else:
            return ResponseData(result=function_lookup_results, status="fail")

    except Exception as e:
        # If something goes wrong, raise an HTTPException
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    process_data(None)
