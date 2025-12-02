import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(out_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("cellscope")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    log_path = os.path.join(out_dir, "cellscope.log")
    err_path = os.path.join(out_dir, "cellscope.error.log")

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(fh)

    eh = RotatingFileHandler(err_path, maxBytes=2 * 1024 * 1024, backupCount=2)
    eh.setFormatter(fmt)
    eh.setLevel(logging.ERROR)
    logger.addHandler(eh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(ch)

    logger.propagate = False
    logger.info("Logger initialized. Logs at %s; errors at %s", log_path, err_path)
    return logger
