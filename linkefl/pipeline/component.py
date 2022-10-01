from abc import ABC, abstractmethod

from linkefl.common.const import Const
from linkefl.crypto.base import CryptoSystem
from linkefl.dataio import NumpyDataset
from linkefl.messenger.base import Messenger
from linkefl.psi.rsa import RSAPSIActive, RSAPSIPassive
from linkefl.vfl.tree import ActiveTreeParty, PassiveTreeParty


class Data:
    data = dict()

    def __new__(cls, **kwargs):
        for key, value in kwargs.items():
            cls.data[key] = value


class Component(ABC):
    def __init__(self, role):
        self.role = role

    @abstractmethod
    def run(self):
        pass


class NumpyDataset_from_csv_ReaderComponent(Component):
    def __init__(self, role, trainset_path, testset_path, dataset_type):
        super().__init__(role)

        self.trainset_path = trainset_path
        self.testset_path = testset_path
        self.dataset_type = dataset_type

    def run(self):
        trainset = NumpyDataset.from_csv(
            role=self.role,
            abs_path=self.trainset_path,
            dataset_type=self.dataset_type,
        )
        testset = NumpyDataset.from_csv(
            role=self.role,
            abs_path=self.testset_path,
            dataset_type=self.dataset_type,
        )

        Data(trainset=trainset, testset=testset)


# class DataReader(Component):
#     def __init__(self, role, func, *args, **kwargs):
#         super().__init__(role)
#
#         self.func = func
#         self.args = args
#         self.kwargs = kwargs
#
#     def run(self):
#         return self.func(self.args, self.kwargs, role=self.role)


class TransformComponent(Component):
    def __init__(self, role, trainset_transform=None, testset_transform=None):
        super().__init__(role)

        self.trainset_transform = trainset_transform
        self.testset_transform = testset_transform

    def run(self):
        if self.trainset_transform is not None:
            trainset = self.trainset_transform(Data.data["trainset"])
        else:
            trainset = Data.data["trainset"]

        if self.testset_transform is not None:
            testset = self.testset_transform(Data.data["testset"])
        else:
            testset = Data.data["testset"]

        Data(trainset=trainset, testset=testset)


class RSAPSIComponent(Component):
    def __init__(self, role, *, messenger, logger, n_processes, crypto_system=None):
        super().__init__(role)

        self.messenger = messenger
        self.logger = logger
        self.n_processes = n_processes
        self.crypto_system = crypto_system

    def run(self):
        trainset = Data.data["trainset"]
        testset = Data.data["testset"]

        if self.role == Const.ACTIVE_NAME:
            trainset_psi = RSAPSIActive(
                ids=trainset.ids,
                messenger=self.messenger,
                cryptosystem=self.crypto_system,
                logger=self.logger,
                num_workers=self.n_processes,
            )
            testset_psi = RSAPSIActive(
                ids=testset.ids,
                messenger=self.messenger,
                cryptosystem=self.crypto_system,
                logger=self.logger,
                num_workers=self.n_processes,
            )
        else:
            trainset_psi = RSAPSIPassive(
                ids=trainset.ids,
                messenger=self.messenger,
                logger=self.logger,
                num_workers=self.n_processes,
            )
            testset_psi = RSAPSIPassive(
                ids=testset.ids,
                messenger=self.messenger,
                logger=self.logger,
                num_workers=self.n_processes,
            )

        trainset_common_ids = trainset_psi.run()
        trainset.filter(trainset_common_ids)
        testset_common_ids = testset_psi.run()
        testset.filter(testset_common_ids)

        Data(trainset=trainset, testset=testset)


class VFLSBTComponent(Component):
    def __init__(self, role, sbt):
        super().__init__(role)

        self.sbt = sbt

    @classmethod
    def active(
        cls,
        n_trees: int,
        task: str,
        n_labels: int,
        crypto_type: str,
        crypto_system: CryptoSystem,
        messenger: Messenger,
        *,
        learning_rate: float = 0.3,
        compress: bool = False,
        max_bin: int = 16,
        max_depth: int = 4,
        reg_lambda: float = 0.1,
        min_split_samples: int = 3,
        min_split_gain: float = 1e-7,
        fix_point_precision: int = 53,
        sampling_method: str = "uniform",
        n_processes: int = 1,
        saving_model: bool = False,
        model_path: str = "./models",
    ):
        active_sbt = ActiveTreeParty(
            n_trees=n_trees,
            task=task,
            n_labels=n_labels,
            crypto_type=crypto_type,
            crypto_system=crypto_system,
            messengers=[messenger],
            learning_rate=learning_rate,
            compress=compress,
            max_bin=max_bin,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            min_split_samples=min_split_samples,
            min_split_gain=min_split_gain,
            fix_point_precision=fix_point_precision,
            sampling_method=sampling_method,
            n_processes=n_processes,
            saving_model=saving_model,
            model_path=model_path,
        )
        return cls(role=Const.ACTIVE_NAME, sbt=active_sbt)

    @classmethod
    def passive(
        cls,
        task: str,
        crypto_type: str,
        messenger: Messenger,
        *,
        max_bin: int = 16,
        saving_model: bool = False,
        model_path: str = "./models",
    ):
        passive_sbt = PassiveTreeParty(
            task=task,
            crypto_type=crypto_type,
            messenger=messenger,
            max_bin=max_bin,
            saving_model=saving_model,
            model_path=model_path,
        )
        return cls(role=Const.PASSIVE_NAME, sbt=passive_sbt)

    def run(self):
        trainset = Data.data["trainset"]
        testset = Data.data["testset"]

        self.sbt.train(trainset, testset)

        Data(model=self.sbt)


class Evaluate(Component):
    def __init__(self):
        pass
