import logging
import os
import sys
from types import SimpleNamespace


class _LineBufferToLogger:
    """
    File-like object that buffers writes and logs complete lines.
    Redirects print() and unhandled tracebacks into logging.
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str):
        if not isinstance(message, str):
            message = str(message)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""

    def isatty(self):
        return False


def configure_program_logging(
    enabled: bool = True,
    log_file: str | None = None,
    level: int = logging.INFO,
    to_console: bool = True,
    append: bool = True,
    capture_prints: bool = True,
    fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
):
    """
    Configure root logging for the whole program.

    - enabled: turn logging on/off without changing call sites
    - log_file: path to log file (None => no file handler)
    - to_console: also emit to terminal (stdout)
    - append: True to append, False to overwrite the file
    - capture_prints: redirect stdout/stderr to logging
    - fmt/datefmt: formatting for all handlers

    Returns a handle with .stop() to undo redirections and remove handlers.
    """

    state = SimpleNamespace(
        logger=None,
        handlers=[],
        old_stdout=None,
        old_stderr=None,
        print_redirectors=[],
        stopped=False,
    )

    if not enabled:
        return state

    logger = logging.getLogger()
    logger.setLevel(level)
    state.logger = logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    if to_console:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        state.handlers.append(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode=("a" if append else "w"), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        state.handlers.append(fh)

    logging.captureWarnings(True)

    # Redirect stdout/stderr -> logging
    if capture_prints:
        stdout_logger = logging.getLogger("STDOUT")
        stderr_logger = logging.getLogger("STDERR")
        out_redirector = _LineBufferToLogger(stdout_logger, logging.INFO)
        err_redirector = _LineBufferToLogger(stderr_logger, logging.ERROR)

        state.old_stdout = sys.stdout
        state.old_stderr = sys.stderr
        sys.stdout = out_redirector
        sys.stderr = err_redirector
        state.print_redirectors = [out_redirector, err_redirector]

    def stop():
        if state.stopped:
            return
        # Flush redirectors
        for r in state.print_redirectors:
            try:
                r.flush()
            except Exception:
                pass

        # Restore std streams
        if state.old_stdout is not None:
            sys.stdout = state.old_stdout
        if state.old_stderr is not None:
            sys.stderr = state.old_stderr

        # Remove handlers
        for h in state.handlers:
            try:
                h.flush()
            except Exception:
                pass
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        logging.captureWarnings(False)
        state.stopped = True

    state.stop = stop
    return state















