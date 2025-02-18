# Copyright (c) Lornatang. All rights reserved.
import logging
import os
from typing import Optional

__all__ = [
    "LOGGER",
]


def configure_logging(name: Optional[str] = None) -> logging.Logger:
    """Configure the logger.

    Args:
        name (Optional[str], optional): The name of the logger. Defaults to None.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    # Get the RANK environment variable, default to -1 if not set
    rank = int(os.getenv("RANK", -1))

    # Set the logging level
    logger_level = logging.INFO if (rank in (-1, 0)) else logging.WARNING
    logger.setLevel(logger_level)

    # Set the logging format
    format_string = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - ddpm - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Create a stream handler for logging to the console
    cli_handler = logging.StreamHandler()
    cli_handler.setFormatter(format_string)
    cli_handler.setLevel(logger_level)
    logger.handlers.clear()
    logger.addHandler(cli_handler)

    return logger


LOGGER = configure_logging(__name__)
