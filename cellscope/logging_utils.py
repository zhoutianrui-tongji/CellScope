import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(out_dir: str, level: str = "INFO") -> logging.Logger:
    out_dir = os.path.realpath(os.path.abspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("cellscope")
    current_out_dir = getattr(logger, "_cellscope_out_dir", None)
    if logger.handlers and current_out_dir == out_dir:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        for handler in logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                handler.setLevel(
                    logging.ERROR
                    if handler.baseFilename.endswith("cellscoper.error.log")
                    else getattr(logging, level.upper(), logging.INFO)
                )
            elif isinstance(handler, logging.StreamHandler):
                handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        return logger
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except OSError:
                pass
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    log_path = os.path.join(out_dir, "cellscoper.log")
    err_path = os.path.join(out_dir, "cellscoper.error.log")

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] CellScoper - %(message)s")

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
    logger._cellscope_out_dir = out_dir
    logger.info(
        "Run logging initialized | log=%s | error_log=%s",
        log_path,
        err_path,
    )
    return logger
