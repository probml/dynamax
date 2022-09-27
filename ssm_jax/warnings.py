# TensorFlow Probability logs a few annoying messages.
# We suppress these by default.
import logging

class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())