import logging
import sys


def enable(module: str, level: int = logging.INFO):
    log = logging.getLogger(module)
    log.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)


def disable(module: str):
    log = logging.getLogger(module)
    log.setLevel(logging.CRITICAL)
    for handler in log.handlers:
        handler.setLevel(logging.CRITICAL)
