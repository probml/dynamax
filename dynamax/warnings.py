"""
TensorFlow Probability logs a few annoying messages. We suppress these by default.
"""
import logging
import warnings


class CheckTypesFilter(logging.Filter):
    """
    Catch "check_types" warnings that are sent to the logger
    """
    def filter(self, record):
        """Filter out check_types warnings"""
        return "check_types" not in record.getMessage()


logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())


# Catch UserWarning: Explicitly requested dtype...
warnings.filterwarnings("ignore", category=UserWarning, message="Explicitly requested dtype")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Using or importing the ABCs")
