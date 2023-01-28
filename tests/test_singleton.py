class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        print(args)
        # print(kwargs)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name, password):
        self.name = name
        self.password = password


if __name__ == "__main__":
    a = Singleton(name="a", password="aaa")
    b = Singleton("b", "bbb")  # override a's attributes
    print(a is b)
