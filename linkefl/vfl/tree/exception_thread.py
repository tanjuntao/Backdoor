import traceback
from threading import Thread


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
    raise Exception("test error")


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
