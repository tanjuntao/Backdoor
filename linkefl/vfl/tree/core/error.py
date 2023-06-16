import traceback
from threading import Thread


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class DisconnectedError(Error):
    """Exception raised for errors in communication.

    Attributes:
        drop_party_id -- disconnected party id
        message -- explanation of the error
    """

    def __init__(self, disconnect_phase: str = "", disconnect_party_id: int = -1):
        self.disconnect_phase = disconnect_phase
        self.disconnect_party_id = disconnect_party_id

    def __str__(self):
        return (
            f"Disconnect Error Message: \ndisconnect_phase: {self.disconnect_phase},"
            f" disconnect_party: passive_party_{self.disconnect_party_id}."
        )


class ExcThread(Thread):
    def run(self):
        self.exception = None
        try:
            if hasattr(self, "_Thread__target"):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(
                    *self._Thread__args, **self._Thread__kwargs
                )
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exception = e
        finally:
            if hasattr(self, "_Thread__target"):
                del self._Thread__target, self._Thread__args, self._Thread__kwargs
            else:
                del self._target, self._args, self._kwargs

    def join(self):
        super(ExcThread, self).join()
        if self.exception:
            raise self.exception
        return self.ret


def f(*args, **kwargs):
    print(args)
    print(kwargs)
    raise Exception("error")


if __name__ == "__main__":
    thread_list = []
    try:
        thread_list.append(ExcThread(target=f, args=(5,), kwargs={"hello": "world"}))
        thread_list.append(ExcThread(target=f, args=(5,), kwargs={"hello": "worlds"}))
        for thread in thread_list:
            thread.daemon = True
            thread.start()
        for thread in thread_list:
            thread.join()
    except BaseException:
        exe_info = traceback.format_exc()
        print(str(exe_info))
