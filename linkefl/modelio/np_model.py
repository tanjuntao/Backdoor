import os
import pickle


class NumpyModelIO:
    def __init__(self):
        pass

    @staticmethod
    def save(model, path, name):
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, name), "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path, name):
        if not os.path.exists(path):
            raise Exception(f"{path} not found.")

        with open(os.path.join(path, name), "rb") as f:
            model = pickle.load(f)

        return model
