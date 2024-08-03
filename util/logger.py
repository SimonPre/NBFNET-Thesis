import datetime
import logging
import os
import traceback
from functools import wraps
from logging.handlers import RotatingFileHandler


class HandledException(Exception):
    pass


class CustomHandler(RotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super(CustomHandler, self).__init__(*args, **kwargs)

    def emit(self, record):
        messages = record.msg.split("\n")
        for message in messages:
            record.msg = message
            super(CustomHandler, self).emit(record)


def create_logger(
    logname, logdir=None, loglevel=logging.INFO, backup_count=5, max_bytes=1_000_000
):
    """
    Creates a logger that also manages archiving and purging of old logs.
    :param logname: name of the log file
    :param logdir: diretory to log to
    :param loglevel: is set to INFO for production by default
    :param backup_count: number of back-ups
    :param max_bytes: maximum size of logfile
    :return: logger instance
    """
    if not logdir:
        logdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, f"{logname}.log")
    level = loglevel

    default_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s %(process)d %(thread)d %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler = CustomHandler(logpath, maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(level)
    handler.setFormatter(default_formatter)

    logger = logging.getLogger(logname)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def log_start_and_finish(func):
    """
    Handy decorator for easy logging of execution time of methods within a class.
    The logger is _must_ be callable via
          self.logger.
    :param func: method to log start and end of execution
    :return:
    """

    @wraps(func)
    def decorator(self, *args, **kwargs):
        try:
            start = datetime.datetime.utcnow()
            self.logger.info("{} started.".format(func.__name__))
            value = func(self, *args, **kwargs)
            took = datetime.datetime.utcnow() - start
            took_sec = took.microseconds * 1e-6 + took.total_seconds()
            self.logger.info(
                "{} finished, took {:.4f} sec.".format(func.__name__, took_sec)
            )
            return value
        except HandledException:
            raise
        except Exception as e:
            traceback.print_exc()
            self.logger.critical("{} finished with error: {}".format(func.__name__, e))
            raise HandledException

    return decorator
