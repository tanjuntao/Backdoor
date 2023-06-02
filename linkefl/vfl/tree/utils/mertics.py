import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from linkefl.vfl.utils.evaluate import Evaluate, Plot


class Metrics:
    def __init__(self, task="binary"):
        self.task = task

        self.train_loss = []
        self.test_loss = []
        self.residual = []

        self.mae = []
        self.mse = []
        self.sse = []
        self.r2 = []

        self.train_acc = []
        self.test_acc = []
        self.train_auc = []
        self.test_auc = []
        self.f1_score = []

    @staticmethod
    def cal_mertics(task, labels, y_preds):
        if task == "regression":
            mae = mean_absolute_error(labels, y_preds)
            mse = mean_squared_error(labels, y_preds)
            sse = mse * len(labels)
            r2 = r2_score(labels, y_preds)
            scores = {"mae": mae, "mse": mse, "sse": sse, "r2": r2}

        elif task == "binary":
            targets = np.round(y_preds).astype(int)
            acc = accuracy_score(labels, targets)
            auc = roc_auc_score(labels, y_preds)
            f1 = f1_score(labels, targets, average="weighted")
            # ks_value, threshold = Evaluate.eval_ks(labels, targets, cut_point=50)
            ks_value, threshold = Evaluate.eval_ks(labels, y_preds, cut_point=50)
            scores = {
                "acc": acc,
                "auc": auc,
                "f1": f1,
                "ks": ks_value,
                "threshold": threshold,
            }

        elif task == "multi":
            targets = np.argmax(y_preds, axis=1)
            acc = accuracy_score(labels, targets)
            scores = {"acc": acc}

        else:
            raise ValueError("No such task label.")

        return scores

    def record_scores(self, scores):
        if "acc" in scores:
            self.test_acc.append(scores["acc"])
        if "auc" in scores:
            self.test_auc.append(scores["auc"])
        if "f1" in scores:
            self.f1_score.append(scores["f1"])
        if "mae" in scores:
            self.mae.append(scores["mae"])
        if "mse" in scores:
            self.mse.append(scores["mse"])
        if "sse" in scores:
            self.sse.append(scores["sse"])
        if "r2" in scores:
            self.r2.append(scores["r2"])

    def record_loss(self, train_loss, test_loss):
        self.train_loss.append(train_loss.mean())
        self.test_loss.append(test_loss.mean())

    def record_residual(self, labels, y_preds):
        self.residual.append((labels - y_preds).mean())

    def record_train_acc_auc(self, train_labels, outputs):
        if self.task == "regression":
            pass
        elif self.task == "binary":
            acc = accuracy_score(train_labels, np.round(outputs).astype(int))
            auc = roc_auc_score(train_labels, outputs)
            self.train_acc.append(acc)
            self.train_auc.append(auc)
        elif self.task == "multi":
            preds = np.argmax(outputs, axis=1)
            acc = accuracy_score(train_labels, preds)
            self.train_acc.append(acc)
        else:
            raise ValueError("No such task label.")

    def save_mertic_pics(self, labels, y_preds, pics_dir):
        # common mertics
        Plot.plot_train_test_loss(
            train_loss=self.train_loss, test_loss=self.test_loss, file_dir=pics_dir
        )

        # specific mertics
        if self.task == "regression":
            Plot.plot_residual(residual=self.residual, file_dir=pics_dir)
            Plot.plot_predict_prob_box(y_prob=y_preds, file_dir=pics_dir)
            Plot.plot_regression_metrics(
                mae=self.mae, mse=self.mse, sse=self.sse, r2=self.r2, file_dir=pics_dir
            )
        elif self.task == "binary":
            Plot.plot_residual(residual=self.residual, file_dir=pics_dir)
            Plot.plot_predict_prob_box(y_prob=y_preds, file_dir=pics_dir)
            Plot.plot_train_test_acc(
                train_acc=self.train_acc, test_acc=self.test_acc, file_dir=pics_dir
            )
            Plot.plot_train_test_auc(
                train_auc=self.train_auc, test_auc=self.test_auc, file_dir=pics_dir
            )
            Plot.plot_f1_score(f1_record=self.f1_score, file_dir=pics_dir)
            Plot.plot_binary_mertics(  # pr, roc, ks, lift
                labels=labels, y_probs=y_preds, cut_point=50, file_dir=pics_dir
            )
            Plot.plot_ordered_lorenz_curve(
                label=labels, y_prob=y_preds, file_dir=pics_dir
            )
            Plot.plot_predict_distribution(y_prob=y_preds, bins=10, file_dir=pics_dir)
        elif self.task == "multi":
            Plot.plot_train_test_acc(
                train_acc=self.train_acc, test_acc=self.test_acc, file_dir=pics_dir
            )
        else:
            raise ValueError("Not supported task.")
