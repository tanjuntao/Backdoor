import datetime
import json
import logging
import os
import pathlib
import queue
import time
from logging.handlers import HTTPHandler, QueueHandler, QueueListener
from typing import Callable, Dict, Optional, Union
from urllib.parse import urlparse

from linkefl.common.const import Const


class GlobalLogger:
    """A singleton wrapper class for Python buildin logger."""

    _logger = None  # object returned by logging.getLogger()
    _http_listener: Optional[QueueListener] = None
    _LOGLEVEL_MAPPER: Dict[str, Callable] = {}
    _PRECISION: int = 6

    def __new__(
        cls,
        *,
        role: str,
        writing_file: bool = False,
        file_path: Optional[str] = None,
        remote_url: Optional[str] = None,
        stacklevel: int = 2,
    ):
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), (
            "role is expected to take from active_party and passive_party, but got"
            f" {role}."
        )

        instance = super().__new__(cls)
        instance.role = role  # type: ignore
        instance.stacklevel = stacklevel  # type: ignore

        # instantiate a Python logger if it does not exist
        if cls._logger is None:
            cls._logger = logging.getLogger("LinkeFL")  # LinkeFL is the logger name
            cls._logger.setLevel(logging.DEBUG)  # set minimum severity level
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s"
            )

            # 1. write logs to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(console_handler)

            # 2. write logs to local file
            if writing_file:
                if file_path is not None:
                    file_dir, file_name = os.path.split(file_path)
                    if not os.path.exists(file_dir):
                        pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
                    full_path = file_path
                else:
                    # ~/.linkefl/ is the cache directory of LinkeFL project
                    logging_dir = os.path.join(pathlib.Path.home(), ".linkefl", "log")
                    # create directories recursively. Same as command 'mkdir -p'
                    # reference: https://stackoverflow.com/a/600612/8418540
                    if not os.path.exists(logging_dir):
                        pathlib.Path(logging_dir).mkdir(parents=True, exist_ok=True)
                    file_name = (
                        datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        + "-"
                        + role
                        + ".log"
                    )
                    full_path = os.path.join(logging_dir, file_name)
                file_handler = logging.FileHandler(full_path, mode="a")  # append
                file_handler.setFormatter(formatter)
                cls._logger.addHandler(file_handler)

            # 3. write logs to a remote http(s) url
            if remote_url is not None:
                parsed_url = urlparse(remote_url)  # return a namedtuple
                host = parsed_url.netloc
                url = parsed_url.path
                http_handler = HTTPHandler(host=host, url=url, method="POST")
                http_handler.setFormatter(formatter)
                # QueueHandler and QueueListener are used for non-blocking http logging
                # initial queue and attach it to QueueHandler
                log_queue = queue.Queue(-1)  # no limit on queue size
                queue_handler = QueueHandler(log_queue)
                # initial QueueListener
                http_listener = QueueListener(log_queue, http_handler)
                cls._http_listener = (
                    http_listener  # make http_listener a class variable
                )
                # attach custom handler to logger
                cls._logger.addHandler(queue_handler)
                # start the listener
                http_listener.start()

            # initial the function mapper of logging severity level
            cls._LOGLEVEL_MAPPER = {
                "debug": cls._logger.debug,
                "info": cls._logger.info,
                "warning": cls._logger.warning,
                "error": cls._logger.error,
                "critical": cls._logger.critical,
            }

        return instance

    def log_metric(
        self,
        epoch: int = 0,
        loss: float = 0.0,
        acc: float = 0.0,
        auc: float = 0.0,
        f1: float = 0.0,
        ks: float = 0.0,
        ks_threshold: float = 0.0,
        mae: float = 0.0,
        mse: float = 0.0,
        sse: float = 0.0,
        r2: float = 0.0,
        total_epoch: int = 0,
        level: str = "info",
    ):
        json_msg = {
            "epoch": epoch,
            "loss": round(float(loss), GlobalLogger._PRECISION),
            "acc": round(float(acc), GlobalLogger._PRECISION),
            "auc": round(float(auc), GlobalLogger._PRECISION),
            "f1": round(float(f1), GlobalLogger._PRECISION),
            "ks": round(float(ks), GlobalLogger._PRECISION),
            "ks_threshold": round(float(ks_threshold), GlobalLogger._PRECISION),
            "mae": round(float(mae), GlobalLogger._PRECISION),
            "mse": round(float(mse), GlobalLogger._PRECISION),
            "sse": round(float(sse), GlobalLogger._PRECISION),
            "r2": round(float(r2), GlobalLogger._PRECISION),
            "time": self._time_formatter(time.time()),
            "progress": epoch / total_epoch,
            "role": self.role,  # type: ignore
        }
        if hasattr(self, "metainfo"):
            json_msg.update(getattr(self, "metainfo"))
        json_msg = json.dumps({"metricLog": json_msg})
        log_func = GlobalLogger._LOGLEVEL_MAPPER[level]
        log_func(json_msg, stacklevel=self.stacklevel)  # type: ignore

    def log_component(
        self,
        name: str,
        status: str,
        begin: float,
        end: Union[float, None],
        duration: float,
        progress: float,
        failure_reason: Optional[str] = None,
        level: str = "info",
    ):
        json_msg = {
            "name": name,
            "status": status,
            "begin": self._time_formatter(begin),
            "end": self._time_formatter(end),
            "duration": duration,
            "progress": progress,
            "failure_reason": failure_reason,
            "role": self.role,  # type: ignore
        }
        if hasattr(self, "metainfo"):
            json_msg.update(getattr(self, "metainfo"))
        json_msg = json.dumps({"componentLog": json_msg})
        log_func = GlobalLogger._LOGLEVEL_MAPPER[level]
        log_func(json_msg, stacklevel=self.stacklevel)  # type: ignore

    def log_step(
        self,
        name: str,
        status: str,
        begin: float,
        end: Union[float, None],
        failure_reason: Optional[str] = None,
        level: str = "info",
    ):
        duration = end - begin
        json_msg = {
            "name": name,
            "status": status,
            "begin": self._time_formatter(begin),
            "end": self._time_formatter(end),
            "duration": duration,
            "failure_reason": failure_reason,
            "role": self.role,  # type: ignore
        }
        if hasattr(self, "metainfo"):
            json_msg.update(getattr(self, "metainfo"))
        json_msg = json.dumps({"stepLog": json_msg})
        log_func = GlobalLogger._LOGLEVEL_MAPPER[level]
        log_func(json_msg, stacklevel=self.stacklevel)  # type: ignore

    def log_task(
        self,
        begin: float,
        end: float,
        status: str,
        failure_reason: Optional[str] = None,
        level: str = "info",
    ):
        json_msg = {
            "begin": self._time_formatter(begin),
            "end": self._time_formatter(end),
            "status": status,
            "failure_reason": failure_reason,
            "role": self.role,  # type: ignore
        }
        if hasattr(self, "metainfo"):
            json_msg.update(getattr(self, "metainfo"))
        json_msg = json.dumps({"taskLog": json_msg})
        log_func = GlobalLogger._LOGLEVEL_MAPPER[level]
        log_func(json_msg, stacklevel=self.stacklevel)  # type: ignore

    def log(
        self,
        content: str,
        level: str = "info",
    ):
        json_msg = {
            "content": content,
            "time": self._time_formatter(time.time()),
            "role": self.role,  # type: ignore
        }
        if hasattr(self, "metainfo"):
            json_msg.update(getattr(self, "metainfo"))
        json_msg = json.dumps({"messageLog": json_msg})
        log_func = GlobalLogger._LOGLEVEL_MAPPER[level]
        log_func(json_msg, stacklevel=self.stacklevel)  # type: ignore

    def close(self):
        if GlobalLogger._http_listener is not None:
            # explicitly call stop() to flush the log records in http queue
            GlobalLogger._http_listener.stop()
            GlobalLogger._http_listener = None
        else:
            pass  # nothing to do

    def get_logger(self):
        return GlobalLogger._logger

    def set_metainfo(self, project_id, flow_id, flow_record_id):
        self.metainfo: dict = {
            "project_id": project_id,
            "flow_id": flow_id,
            "flow_record_id": flow_record_id,
        }

    def _time_formatter(self, timestamp):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


if __name__ == "__main__":
    logger1 = GlobalLogger(role=Const.ACTIVE_NAME, writing_file=True)
    logger2 = GlobalLogger(role=Const.ACTIVE_NAME, writing_file=True)

    print(id(logger1), id(logger2))  # different
    print(id(logger1.get_logger()), id(logger2.get_logger()))  # same

    logger1.log("hello world from logger1")
    logger2.log("nihao from logger2")

    logger1.close()
    logger2.close()
