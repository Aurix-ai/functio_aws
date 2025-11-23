import subprocess
import sys
import time

# Global variable to store the server process
_server_process = None

def start_server(clickhouse_server_path: str = "/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse"):
    """
    Start the ClickHouse server in a separate process.
    If a server is already running, stop it first before starting a new one.
    
    Args:
        clickhouse_server_path: Path to the clickhouse server executable
        
    Returns:
        subprocess.Popen: The server process object
    """
    global _server_process
    
    if _server_process is not None:
        # Server already running - stop it first
        stop_server()
    
    # Start the server process (runs in background)
    _server_process = subprocess.Popen(
        [clickhouse_server_path, "server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a bit for server to start up
    time.sleep(2)
    
    return _server_process

def execute_query(query: str, clickhouse_client_path: str = "/home/ubuntu/ClickHouse_debug/build_debug/programs/clickhouse"):
    """
    Execute a ClickHouse query using the client against the running server.
    The server is started automatically if not already running.
    
    Args:
        query: The SQL query string to execute
        clickhouse_client_path: Path to the clickhouse client executable
        
    Returns:
        tuple: (success: bool, output: str, error: str)
    """
    global _server_process
    
    # Ensure server is running
    if _server_process is None:
        start_server()
    
    try:
        # Run clickhouse client with the query
        result = subprocess.run(
            [clickhouse_client_path, "client", "--query", query],
            capture_output=True,
            text=True,
            check=False,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "", "Query execution timed out after 5 minutes"
    except Exception as e:
        return False, "", f"Error executing query: {str(e)}"

def stop_server():
    """
    Stop the ClickHouse server process.
    """
    global _server_process
    
    if _server_process is not None:
        _server_process.terminate()
        _server_process.wait()
        _server_process = None

    result = subprocess.run(
            ["pidof", "clickhouse"],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode == 0 and result.stdout.strip():
        # Successfully found running clickhouse process(es) - kill them
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                subprocess.run(["sudo", "kill", "-9", pid], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to kill existing clickhouse process {pid}: {e}", file=sys.stderr)
        time.sleep(3)  # Wait for process(es) to be killed