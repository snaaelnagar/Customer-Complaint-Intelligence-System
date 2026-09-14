import logging
import os
from logging.handlers import RotatingFileHandler


LOG_DIR = "logs"
LOG_FILE = "app.log"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)


def get_logger(name: str = "customer_complaint_intelligence") -> logging.Logger:
    """
    Create and return application logger.

    This logger writes logs to both:
    1. Console - for live debugging
    2. File - for long-term debugging

    RotatingFileHandler is used so log files do not grow endlessly.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs when file is imported multiple times
    if logger.handlers:
        return logger

    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    file_handler = RotatingFileHandler(
        filename=LOG_PATH,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()


if __name__ == "__main__":
    logger.info("Logger test started")
    logger.info("This is an info message")
    logger.warning("This is a warning message")

    try:
        result = 10 / 0
    except Exception as e:
        logger.exception("Logger exception test failed")

    logger.info("Logger test completed")