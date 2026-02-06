from typing import Type, Literal
import logging
import traceback
import warnings


class LogColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # Reset color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PipelineFormatter(logging.Formatter):
    """ Custom logging formatter that adds colors based on log level """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        log_colors = {
            logging.DEBUG: LogColors.OKCYAN,
            logging.INFO: LogColors.OKGREEN,
            logging.WARNING: LogColors.WARNING,
            logging.ERROR: LogColors.FAIL,
            logging.CRITICAL: LogColors.BOLD + LogColors.FAIL,
        }
        color = log_colors.get(record.levelno, LogColors.ENDC)
        message = super().format(record)
        return f"{color}{message}{LogColors.ENDC}"

class SystemFormatter(logging.Formatter):
    """ Custom logging formatter that adds colors based on log level """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        log_colors = {
            logging.DEBUG: LogColors.OKCYAN,
            logging.INFO: LogColors.OKCYAN,
            logging.WARNING: LogColors.WARNING,
            logging.ERROR: LogColors.FAIL,
            logging.CRITICAL: LogColors.BOLD + LogColors.FAIL,
        }
        color = log_colors.get(record.levelno, LogColors.ENDC)
        message = super().format(record)
        return f"{color}{message}{LogColors.ENDC}"


class ModuleFormatter(logging.Formatter):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        log_colors = {
            logging.DEBUG: LogColors.OKCYAN,
            logging.INFO: LogColors.OKBLUE,
            logging.WARNING: LogColors.WARNING,
            logging.ERROR: LogColors.FAIL,
            logging.CRITICAL: LogColors.BOLD + LogColors.FAIL,
        }
        color = log_colors.get(record.levelno, LogColors.ENDC)
        message = super().format(record)
        return f"{color}{message}{LogColors.ENDC}"


class Formatter():
    @classmethod
    def get_formatter(cls, formatter: str) -> Type[logging.Formatter]:
        if formatter == "pipeline":
            return PipelineFormatter(PipelineFormatter.log_format)
        elif formatter == "system":
            return SystemFormatter(SystemFormatter.log_format)
        elif formatter == "module":
            return ModuleFormatter(ModuleFormatter.log_format)
        else:
            raise ValueError(f"Invalid formatter: {formatter}")


def setup_logging(
    name=__name__,
    verbose: str="INFO",
    formatter: Literal["system", "pipeline", "module"] = "system"
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(verbose)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(verbose)
    handler.setFormatter(Formatter.get_formatter(formatter))

    logger.addHandler(handler)
    return logger

def log_exception(logger, message: str):
    """
    Logs an error message along with the file name and line number where the error occurred.
    """
    exc_type, exc_value, exc_traceback = traceback.sys.exc_info()
    tb = traceback.extract_tb(exc_traceback)
    if tb:
        # Get the last entry from the traceback
        filename, line, func, text = tb[-1]
        logger.error(f"{message} - Exception occurred in {filename}, line {line}: {exc_value}")
    else:
        # Fallback to a general error log if no traceback is found
        logger.error(f"{message} - {exc_value}")


warnings.filterwarnings("ignore", "DeprecationWarning.*")

def set_deprecation_warnings(enabled: bool):
    """
    Enable or disable deprecation warnings globally for the package.

    Args:
        enabled (bool): If True, deprecation warnings will be shown.
    """
    if enabled:
        warnings.filterwarnings("default", "DeprecationWarning.*")
    else:
        warnings.filterwarnings("ignore", "DeprecationWarning.*")