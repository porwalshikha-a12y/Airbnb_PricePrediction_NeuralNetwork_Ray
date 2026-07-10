# logger.py

import logging
import os


def get_logger(name="airbnb_project", log_file="app.log", level=logging.INFO):
    """
    Creates and returns a configured logger.

    Args:
        name (str): Logger name (usually __name__ from caller)
        log_file (str): File to store logs
        level: Logging level (INFO, DEBUG, etc.)

    Returns:
        logging.Logger object
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers (important when importing in multiple files)
    if logger.handlers:
        return logger

    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    # Format for logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler (prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler (writes to file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger