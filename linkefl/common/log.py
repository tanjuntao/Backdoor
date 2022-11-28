import datetime
import json
import logging
import os
import pathlib
import queue
import time
from logging.handlers import HTTPHandler, QueueHandler, QueueListener

from linkefl.common.const import Const


class GlobalLogger:
    """This class is a Python logger singleton."""
    _logger = None
    _http_listener = None
    _loglevel_dict = {}
    FLOAT_PRECISION = 6

    def __new__(cls, *,
                role,
                writing_file=False,
                writing_http=False,
                http_host=None, http_port=None, http_url=None):
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), 'invalid role.'
        if writing_http:
            if None in (http_host, http_port, http_url):
                raise ValueError('http host/port/url should not be None.')

        instance = super().__new__(cls)
        instance.role = role
        instance.writing_http = writing_http

        # generat Python logger if it does not exist
        if cls._logger is None:
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
                file_handler = logging.FileHandler(full_path, mode='a') # append
                file_handler.setFormatter(formatter)
                cls._logger.addHandler(file_handler)

            # 3. write logging message to a remote http(s) server
            if writing_http:
                # initial http handler, subustite the host and url if needed.
                http_handler = HTTPHandler(
                    host=http_host + ':' + str(http_port),
                    url=http_url,
                    method='POST'
                )
                http_handler.setFormatter(formatter)
                # QueueHandler and QueueListener are used for non-blocking http logger
                # initial queue and attach it to QueueHandler
                log_queue = queue.Queue(-1) # no limit on queue size
                queue_handler = QueueHandler(log_queue)
                # initial QueueListener
                http_listener = QueueListener(log_queue, http_handler)
                cls._http_listener = http_listener # make http_listener an class variable
                # attach custom handler to logger
                cls._logger.addHandler(queue_handler)
                # start the listener
                http_listener.start()

            cls._loglevel_dict = {
                Const.DEBUG: cls._logger.debug,
                Const.INFO: cls._logger.info,
                Const.WARNING: cls._logger.warning,
                Const.ERROR: cls._logger.error,
                Const.CRITICAL: cls._logger.critical,
            }

        return instance

    def log_metric(self, epoch, loss, acc, auc, f1, total_epoch, level='info'):
        json_msg = json.dumps({
            'metricLog': {
                'epoch': epoch + 1,
                'loss': round(loss, GlobalLogger.FLOAT_PRECISION),
                'acc': round(acc, GlobalLogger.FLOAT_PRECISION),
                'auc': round(auc, GlobalLogger.FLOAT_PRECISION),
                'f1': round(f1, GlobalLogger.FLOAT_PRECISION),
                'time': self.time_formatter(time.time()),
                'progress': (epoch + 1) / total_epoch,
                'role': self.role,
            }
        })
        log_func = GlobalLogger._loglevel_dict[level]
        log_func(json_msg)

    def log_component(self, name, status,
                      begin, end, duration, progress,
                      failure_reason=None, level='info'):
        json_msg = json.dumps({
            'componentLog': {
                'name': name,
                'status': status,
                'begin': self.time_formatter(begin),
                'end': self.time_formatter(end),
                'duration': duration,
                'progress': progress,
                'failure_reason': failure_reason,
                'role': self.role,
            }
        })
        log_func = GlobalLogger._loglevel_dict[level]
        log_func(json_msg)

    def log_task(self, begin, end, status, level='info'):
        json_msg = json.dumps({
            'taskLog': {
                'begin': self.time_formatter(begin),
                'end': self.time_formatter(end),
                'status': status,
                'role': self.role,
            }
        })
        log_func = GlobalLogger._loglevel_dict[level]
        log_func(json_msg)

    def log(self, content, level='info'):
        json_msg = json.dumps({
            'messageLog': {
                'content': content,
                'time': self.time_formatter(time.time()),
                'role': self.role,
            }
        })
        log_func = GlobalLogger._loglevel_dict[level]
        log_func(json_msg)

    def close(self):
        if GlobalLogger._http_listener is not None:
            # you must call stop() explicitly to flush the records in the queue
            GlobalLogger._http_listener.stop()
            GlobalLogger._http_listener = None
        else:
            pass # nothing to do

    def get_logger(self):
        return GlobalLogger._logger

    def time_formatter(self, timestamp):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))


if __name__ == '__main__':
    logger1 = GlobalLogger(role=Const.ACTIVE_NAME, writing_file=True, writing_http=False)
    logger2 = GlobalLogger(role=Const.ACTIVE_NAME, writing_file=True, writing_http=False)

    # logger1 = GlobalLogger(role=Const.ACTIVE_NAME,
    #                        writing_file=True,
    #                        writing_http=True,
    #                        http_host='127.0.0.1',
    #                        http_port=5000,
    #                        http_url='/log')
    # logger2 = GlobalLogger(role=Const.ACTIVE_NAME,
    #                        writing_file=True,
    #                        writing_http=True,
    #                        http_host='127.0.0.1',
    #                        http_port=5000,
    #                        http_url='/log')

    print(id(logger1), id(logger2)) # different
    print(id(logger1.get_logger()), id(logger2.get_logger())) # same

    logger1.log('hello world from logger1')
    logger2.log('nihao from logger2')

    logger1.close()
    logger2.close()