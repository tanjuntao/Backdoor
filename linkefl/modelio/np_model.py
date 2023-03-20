import os
import pathlib
import pickle


class NumpyModelIO:
    def __init__(self):
        pass

    @staticmethod
    def save(model, model_dir, model_name):
        if not os.path.exists(model_dir):
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(model_dir, model_name), "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(model_dir, model_name):
        if not os.path.exists(model_dir):
            raise Exception(f"{model_dir} not found.")

        with open(os.path.join(model_dir, model_name), "rb") as f:
            model = pickle.load(f)

        return model
