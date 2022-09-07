import datetime
import logging
from logging.handlers import HTTPHandler, QueueHandler, QueueListener
import os
import pathlib
import queue

from linkefl.common.const import Const


class GlobalLogger:
    """This class is a Python logger singleton."""
    _logger = None

    def __new__(cls, role, writing_file=False, writing_http=False):
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'invalid role.'

        if cls._logger is None:
            cls._logger = super().__new__(cls) # like super().__init__(self)
            cls._logger = logging.getLogger('LinkeFL') # LinkeFL is the logger name
            cls._logger.setLevel(logging.DEBUG) # minimum logger severity level
            formatter = logging.Formatter('%(asctime)s [%(levelname)s | '
                                          '%(filename)s:%(lineno)s] > %(message)s')

            # 1. write logging message to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(console_handler)

            # 2. write logging message to disk file
            if writing_file:
                # .linkefl is the project cache directory
                logging_dir = os.path.join(pathlib.Path.home(), '.linkefl', 'log')
                # create directories recursively. Same as command 'mkdir -p'
                # reference: https://stackoverflow.com/a/600612/8418540
                if not os.path.exists(logging_dir):
                    pathlib.Path(logging_dir).mkdir(parents=True, exist_ok=True)
                file_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S") +\
                            "-" + role + ".log"
                full_path = os.path.join(logging_dir, file_name)
                file_handler = logging.FileHandler(full_path)
                file_handler.setFormatter(formatter)
                cls._logger.addHandler(file_handler)

            # 3. write logging message to a remote http(s) server
            if writing_http:
                # initial http handler, subustite the host and url if needed.
                http_handler = HTTPHandler(
                    host='localhost:5000',
                    url='/log',
                    method='POST'
                )
                http_handler.setFormatter(formatter)

                # QueueHandler and QueueListener are used for non-blocking http logger
                # initial queue and attach it to QueueHandler
                log_queue = queue.Queue(-1) # no limit on queue size
                queue_handler = QueueHandler(log_queue)
                # initial QueueListener
                http_listener = QueueListener(log_queue, http_handler)
                # attach custom handler to logger
                cls._logger.addHandler(queue_handler)
                # start the listener
                http_listener.start()

        return cls._logger


if __name__ == '__main__':
    logger = GlobalLogger(role=Const.ACTIVE_NAME, writing_file=True)
    logger.info('INFO-hello world')

    logger = GlobalLogger(role=Const.ACTIVE_NAME, writing_file=True)
    logger.warning('Warning-Hello world')

