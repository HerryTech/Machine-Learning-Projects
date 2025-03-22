# src/logger.py
import logging
import os

# Create a logs folder if it doesn’t exist
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "app.log"),  # Save logs in a file
    level=logging.INFO,  # Only track important logs (INFO and ERROR)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name):
    """Create a logger with the given name."""
    return logging.getLogger(name)
