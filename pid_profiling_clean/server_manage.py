import subprocess
import socket
import time
import uuid
from config import SERVER_HOST, SERVER_PORT, CPP_APP_EXECUTABLE
import logging

class DBLifecycleManager:
    def init(self, start_cmd, host, port):
        self.start_cmd = start_cmd
        self.host = host
        self.port = port
        self.process = None
        self.session_id = None # Unique ID for the current "run"
    
    def _is_port_open(self):
        """Checks if the DB is actually listening for connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def ensure_up(self):
        """
        The Magic Function.
        Checks if server is up. If not, starts it.
        Returns: current_session_id
        """
        # 1. Check if process is known and running
        if self.process and self.process.poll() is None:
            # 2. Double check if port is actually answering
            if self._is_port_open():
                return self.session_id
            else:
                logging.warning("Process exists but port is closed. It might be hanging/starting.")
                # Logic to wait or kill/restart could go here
                
        # If we get here, it's down. Start it.
        logging.info('Starting DB Server')
        self.process = subprocess.Popen(
            self.start_cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        
        # Wait for "Green Light"
        retries = 0
        while not self._is_port_open():
            time.sleep(0.5)
            retries += 1
            if retries > 20: # 10 seconds timeout
                self.process.kill()
                raise RuntimeError("DB Server failed to launch.")
        
        # Assign a new Session ID because it's a fresh instance
        self.session_id = str(uuid.uuid4())
        logging.info(f"DB Server started")
        return self.session_id

    def shutdown(self):
        """Gracefully stops the server."""
        if self.process:
            logging.info(">>> [Manager] Shutting down DB Server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.session_id = None
            logging.info(">>> [Manager] DB Server stopped.")

    def get_status(self):
        """Returns a dict useful for logic checks."""
        alive = self.process is not None and self.process.poll() is None and self._is_port_open()
        return {
            "is_alive": alive,
            "session_id": self.session_id
        }

db_controller = DBLifecycleManager(start_cmd=[CPP_APP_EXECUTABLE, "server"], host=SERVER_HOST, port=SERVER_PORT)