from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fl_analyze_unified import unified_strategy
import logging

# 1. Create a Pydantic model for request data validation
# This ensures the incoming JSON has a 'value' field of type float.
class RequestData(BaseModel):
    value: str

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
    Receives data, processes it, and returns the result.
    """
    try:
        # Delete this later
        #data = '/home/ubuntu/ClickHouse/build_debug/programs/clickhouse local --query "SELECT quantilesTDigest(0.5,0.9,0.99)(toFloat64(sin(number)) * (number % 1000)) FROM numbers_mt(30000000);"'
        print(f"Received data: {data}")
        #you can access the received data by data.value
        result_value = unified_strategy(data.value, "report", False, True, ["/home/ubuntu/ClickHouse/contrib"], prefix_path="/home/ubuntu/ClickHouse", custom_prefix="/home/rocky/ClickHouse_functio")
        #result_value is a list of tuples [(function_name, full_path), ...]
        logging.info('FINISHED')
        return ResponseData(result=result_value, status="success")

    except Exception as e:
        # If something goes wrong, raise an HTTPException
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    process_data(None)
