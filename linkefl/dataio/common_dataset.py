from __future__ import annotations  # python >= 3.7, give type hint before definition

import io
import random
from typing import Union

import numpy as np
import pandas as pd
import torch

from linkefl.base import BaseTransformComponent
from linkefl.common.const import Const


class CommonDataset:
    """Common Dataset"""

    mappings = None  # mappings for pandas non-numeric columns

    def __init__(
        self,
        role: str,
        raw_dataset: Union[np.ndarray, torch.Tensor],
        header: list,
        dataset_type: str,
        transform: BaseTransformComponent = None,
        header_type=None,
    ):
        super(CommonDataset, self).__init__()
        assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), "Invalid role"
        self.role = role
        self._header = header
        self.header_type = header_type
        self.dataset_type = dataset_type
        self.has_label = True if role == Const.ACTIVE_NAME else False

        if transform is not None:
            raw_dataset = transform.fit(raw_dataset, role=role)

        # after this function call, self._raw_dataset attribute will be created
        self.set_dataset(raw_dataset)

    @classmethod
    def train_test_split(
        cls,
        dataset: CommonDataset,
        test_size: float,
        option: str = Const.SEQUENCE,
        seed=None,
    ):
        """Split the whole dataset into trainset and testset"""
        assert isinstance(
            dataset, CommonDataset
        ), "dataset should be an instance of CommonDataset"
        assert 0 <= test_size < 1, "test size should be in range [0, 1)"

        if option == Const.SEQUENCE:
            perm = list(range(dataset.n_samples))
        elif option == Const.RANDOM:
            if seed is not None:
                random.seed(seed)
            perm = list(range(dataset.n_samples))
            random.shuffle(perm)
        else:
            raise ValueError(
                "in train_test_split method, the option of sample "
                "permutation can only take from SEQUENCE AND RANDOM."
            )

        n_train_samples = int(dataset.n_samples * (1 - test_size))
        raw_trainset = dataset.get_dataset()[perm[:n_train_samples], :]
        raw_testset = dataset.get_dataset()[perm[n_train_samples:], :]

        trainset = cls(
            role=dataset.role,
            raw_dataset=raw_trainset,
            header=dataset.header,
            dataset_type=dataset.dataset_type,
        )
        testset = cls(
            role=dataset.role,
            raw_dataset=raw_testset,
            header=dataset.header,
            dataset_type=dataset.dataset_type,
        )

        return trainset, testset

    @classmethod
    def feature_split(
        cls,
        dataset: CommonDataset,
        n_splits: int,
        option: str = Const.SEQUENCE,
        seed=None,
    ):
        assert isinstance(
            dataset, CommonDataset
        ), "dataset should be an instance of CommonDataset"
        if option == Const.SEQUENCE:
            perm = list(range(dataset.n_features))
        elif option == Const.RANDOM:
            if seed is not None:
                random.seed(seed)
            perm = list(range(dataset.n_features))
            random.shuffle(perm)
        else:
            raise ValueError(
                "in feature_split method, the option of feature "
                "permutation can only take from SEQUENCE AND RANDOM."
            )

        offset = 2 if dataset.role == Const.ACTIVE_NAME else 1
        permed_features = dataset.features[:, perm]
        permed_header = np.array(dataset.header[offset:])[perm].tolist()
        ids = dataset.ids  # is a Python list
        step = dataset.n_features // n_splits

        splitted_datasets = []
        for i in range(n_splits):
            begin_idx = i * step
            if i != n_splits - 1:
                end_idx = (i + 1) * step
            else:
                end_idx = dataset.n_features

            splitted_feats = permed_features[:, begin_idx:end_idx]
            splitted_header = permed_header[begin_idx:end_idx]
            if isinstance(splitted_feats, np.ndarray):  # NumPy Array
                raw_dataset = np.concatenate(
                    (np.array(ids)[:, np.newaxis], splitted_feats), axis=1
                )
            else:  # PyTorch Tensor
                raw_dataset = torch.cat(
                    (torch.unsqueeze(torch.tensor(ids), 1), splitted_feats), dim=1
                )
            splitted_header = dataset.header[:offset] + splitted_header
            curr_dataset = cls(
                role=dataset.role,
                raw_dataset=raw_dataset,
                header=splitted_header,
                dataset_type=dataset.dataset_type,
            )
            splitted_datasets.append(curr_dataset)

        return splitted_datasets

    @classmethod
    def buildin_dataset(
        cls,
        role,
        dataset_name,
        root,
        train,
        passive_feat_frac,
        feat_perm_option,
        download=False,
        transform=None,
        seed=None,
    ):
        def _check_params():
            assert role in (Const.ACTIVE_NAME, Const.PASSIVE_NAME), "Invalid role"
            assert (
                dataset_name in Const.BUILDIN_DATASETS
            ), "not supported dataset right now"
            assert (
                0 <= passive_feat_frac < 1
            ), "passive_feat_frac should be in range [0, 1)"
            assert feat_perm_option in (
                Const.RANDOM,
                Const.SEQUENCE,
                Const.IMPORTANCE,
            ), "invalid feat_perm_option, please check it again"

        # function body
        _check_params()
        np_dataset, header = cls._load_buildin_dataset(
            role=role,
            name=dataset_name,
            root=root,
            train=train,
            download=download,
            frac=passive_feat_frac,
            perm_option=feat_perm_option,
            seed=seed,
        )
        if dataset_name in Const.DATA_TYPE_DICT[Const.REGRESSION]:
            dataset_type = Const.REGRESSION
        else:
            dataset_type = Const.CLASSIFICATION

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=header,
            dataset_type=dataset_type,
            transform=transform,
        )

    @classmethod
    def dummy_dataset(
        cls,
        role,
        dataset_type,
        n_samples,
        n_features,
        passive_feat_frac,
        transform=None,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)
        _ids = np.arange(n_samples)
        _feats = [random.random() for _ in range(n_samples * n_features)]
        _feats = np.array(_feats).reshape(n_samples, n_features)
        _feats_header = ["x{}".format(i) for i in range(n_features)]

        num_passive_feats = int(passive_feat_frac * n_features)
        if role == Const.PASSIVE_NAME:
            np_dataset = np.concatenate(
                (_ids[:, np.newaxis], _feats[:, :num_passive_feats]), axis=1
            )
            header = ["id"] + _feats_header[:num_passive_feats]
        else:
            if dataset_type == Const.CLASSIFICATION:
                _labels = np.array([random.choice([0, 1]) for _ in range(n_samples)])
            else:
                _labels = np.array([random.random() for _ in range(n_samples)])
            np_dataset = np.concatenate(
                (
                    _ids[:, np.newaxis],
                    _labels[:, np.newaxis],
                    _feats[:, num_passive_feats:],
                ),
                axis=1,
            )
            header = ["id"] + ["y"] + _feats_header[num_passive_feats:]

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=header,
            dataset_type=dataset_type,
            transform=transform,
        )

    @classmethod
    def from_mysql(
        cls,
        role,
        dataset_type,
        host,
        user,
        password,
        database,
        table,
        *,
        target_fields=None,
        excluding_fields=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
        port=3306,
    ):
        """Load dataset from MySQL database."""
        import pymysql

        connection = pymysql.connect(
            host=host,
            user=user,
            port=port,
            password=password,
            database=database,
            cursorclass=pymysql.cursors.DictCursor,
        )
        with connection:
            with connection.cursor() as cursor:
                selected_fields = cls._get_selected_fields(
                    db_type="mysql",
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields,
                )
                sql = (
                    "select"
                    + " "
                    + ",".join(selected_fields)
                    + " "
                    + "from {}".format(table)
                )
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)
                header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]

        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(
            df_dataset, row_threshold=row_threshold, column_threshold=column_threshold
        )
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=selected_fields,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_mariadb(
        cls,
        role,
        dataset_type,
        host,
        user,
        password,
        database,
        table,
        *,
        target_fields=None,
        excluding_fields=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
        port=3306,
    ):
        """
        Load dataset from MariaDB database.
        Note that the default port is 3306, the same as mysql
        """
        import mariadb

        connection = mariadb.connect(
            host=host, user=user, port=port, password=password, database=database
        )

        with connection:
            with connection.cursor() as cursor:
                selected_fields = cls._get_selected_fields(
                    db_type="mariadb",
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields,
                )
                sql = (
                    "select"
                    + " "
                    + ",".join(selected_fields)
                    + " "
                    + "from {}".format(table)
                )
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)
                header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]

        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(
            df_dataset, row_threshold=row_threshold, column_threshold=column_threshold
        )
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=selected_fields,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_oracle(
        cls,
        role,
        dataset_type,
        host,
        user,
        password,
        database,
        table,
        *,
        target_fields=None,
        excluding_fields=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
        port=1521,
    ):
        import cx_Oracle

        service_name = database
        dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
        connection = cx_Oracle.connect(
            user=user, password=password, dsn=dsn, encoding="UTF-8"
        )
        with connection:
            with connection.cursor() as cursor:
                selected_fields = cls._get_selected_fields(
                    db_type="oracle",
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields,
                )
                sql = (
                    "select"
                    + " "
                    + ",".join(selected_fields)
                    + " "
                    + "from {}".format(table)
                )
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)
                header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]

        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(
            df_dataset, row_threshold=row_threshold, column_threshold=column_threshold
        )
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=selected_fields,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_gaussdb(
        cls,
        role,
        dataset_type,
        host,
        user,
        password,
        database,
        table,
        *,
        target_fields=None,
        excluding_fields=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
        port=6789,
    ):
        """Load dataset from Gaussdb database."""
        # Note that Python is a dynamic programming language, so this error message can
        # be safely ignored in case where you do not need to call this method with a
        # Python interpreter without psyconpg2 package installed. If you do get
        # annoyed by this error message, you can use pip3 to install the psyconpg2
        # package to suppress it. But the package installed via pip3 may be incompatible
        # when connnecting to gaussdb, this is why it is not included in LinkeFL's
        # requirements. If your application needs to load data from gaussdb, it's
        # required that you should first install guassdb manually and generate psycopg2
        # package which can be imported by a Python script, and then use this method to
        # load raw data from guassdb into LinkeFL project.
        import psycopg2

        connection = psycopg2.connect(
            database=database, user=user, password=password, host=host, port=port
        )
        with connection:
            with connection.cursor() as cursor:
                selected_fields = cls._get_selected_fields(
                    db_type="gaussdb",
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields,
                )
                sql = (
                    "select"
                    + " "
                    + ",".join(selected_fields)
                    + " "
                    + "from {}".format(table)
                )
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)
                header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]

        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(
            df_dataset, row_threshold=row_threshold, column_threshold=column_threshold
        )
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=selected_fields,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_gbase8a(
        cls,
        role,
        dataset_type,
        host,
        user,
        password,
        database,
        table,
        *,
        target_fields=None,
        excluding_fields=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
        port=6789,
    ):
        """Load dataset from gbase8a database."""
        import pymysql

        connection = pymysql.connect(
            database=database, user=user, password=password, host=host, port=port
        )
        with connection:
            with connection.cursor() as cursor:
                selected_fields = cls._get_selected_fields(
                    db_type="gbase8a",
                    cursor=cursor,
                    table=table,
                    target_fields=target_fields,
                    excluding_fields=excluding_fields,
                )
                sql = (
                    "select"
                    + " "
                    + ",".join(selected_fields)
                    + " "
                    + "from {}".format(table)
                )
                cursor.execute(sql)
                results = cursor.fetchall()
                df_dataset = pd.DataFrame.from_dict(results)
                header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]

        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(
            df_dataset, row_threshold=row_threshold, column_threshold=column_threshold
        )
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=selected_fields,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_db2(
        cls,
        role,
        dataset_type,
        host,
        user,
        password,
        database,
        table,
        *,
        target_fields=None,
        excluding_fields=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
        port=None,
    ):
        """
        Load dataset from IBM DB2 database.
        No default port
        """
        import ibm_db_dbi

        connection = ibm_db_dbi.connect(database, user, password)

        selected_fields = cls._get_selected_fields(
            db_type="db2",
            cursor=None,
            table=table,
            target_fields=target_fields,
            excluding_fields=excluding_fields,
            conn=connection,
        )
        sql = "select" + " " + ",".join(selected_fields) + " " + "from {}".format(table)

        df_dataset = pd.read_sql(sql, connection)
        header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]
        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(
            df_dataset, row_threshold=row_threshold, column_threshold=column_threshold
        )
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=selected_fields,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_csv(
        cls,
        role,
        abs_path,
        dataset_type,
        delimiter=",",
        has_header=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
    ):
        header_arg = 0 if has_header else None
        df_dataset = pd.read_csv(
            abs_path,
            delimiter=delimiter,
            header=header_arg,
            skipinitialspace=True,  # skip spaces after delimiter
        )
        header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]

        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(
            df_dataset, row_threshold=row_threshold, column_threshold=column_threshold
        )
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        if has_header:
            header = df_dataset.columns.values.tolist()
        else:
            offset = 1 if role == Const.PASSIVE_NAME else 2
            n_feats = np_dataset.shape[1] - offset
            header = cls._gen_header(role, n_feats)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=header,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_excel(
        cls,
        role,
        abs_path,
        dataset_type,
        has_header=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
    ):
        """Load dataset from excel.
        need dependency package openpyxl, support .xls .xlsx
        """
        header_arg = 0 if has_header else None
        df_dataset = pd.read_excel(abs_path, header=header_arg, index_col=False)

        header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]
        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(
            df_dataset, row_threshold=row_threshold, column_threshold=column_threshold
        )
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        if has_header:
            header = df_dataset.columns.values.tolist()
        else:
            offset = 1 if role == Const.PASSIVE_NAME else 2
            n_feats = np_dataset.shape[1] - offset
            header = cls._gen_header(role, n_feats)

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=header,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_json(
        cls,
        role,
        abs_path,
        dataset_type,
        data_field="data",
        has_header=True,
        existing_json=None,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
    ):
        if existing_json is None:
            whole_json = pd.read_json(abs_path)
            raw_data = whole_json[data_field].tolist()  # a Python list
        else:
            whole_json = existing_json
            raw_data = whole_json[data_field]  # a Python list

        df_dataset = pd.DataFrame.from_dict(raw_data)
        header_type = [str(_type) for _type in df_dataset.dtypes.tolist()]
        df_dataset = cls._date_data(df_dataset, columns=date_columns)
        df_dataset = cls._clean_data(df_dataset, row_threshold, column_threshold)
        df_dataset = cls._outlier_data(df_dataset, role=role)
        df_dataset = cls._fill_data(df_dataset)
        np_dataset = cls._pandas2numpy(df_dataset, mappings=mappings)

        header = list(raw_data[0].keys())  # each element in raw_data is a dict

        return cls(
            role=role,
            raw_dataset=np_dataset,
            header=header,
            dataset_type=dataset_type,
            transform=transform,
            header_type=header_type,
        )

    @classmethod
    def from_api(
        cls,
        role,
        url,
        dataset_type,
        post_params=None,
        data_field="data",
        has_header=True,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
    ):
        import json

        import requests

        resp = requests.post(url=url, json=post_params)
        existing_json = json.loads(resp.text)

        return cls.from_json(
            role=role,
            abs_path=None,
            dataset_type=dataset_type,
            data_field=data_field,
            existing_json=existing_json,
            date_columns=date_columns,
            row_threshold=row_threshold,
            column_threshold=column_threshold,
            mappings=mappings,
            transform=transform,
        )

    @classmethod
    def from_anyfile(
        cls,
        role,
        abs_path,
        dataset_type,
        is_local,
        has_header=False,
        date_columns=None,
        row_threshold=0.3,
        column_threshold=0.3,
        mappings=None,
        transform=None,
    ):
        extension = abs_path.split(".")[-1]
        # if the data file is in remote
        if not is_local:
            import requests

            # abs_path is an url, e.g., http://10.10.10.81:8001/digits_active.json
            data_raw = requests.get(abs_path).content
            # abs_path is now a StringIO object
            abs_path = io.StringIO(data_raw.decode("utf-8"))

        if extension in ("csv", "txt", "dat"):
            # TODO: parse delimiter for both local file and remote file
            delimiter = ","
            # read first two lines to determine the delimiter
            # with open(abs_path) as f:
            #     first_line = f.readline()
            #     second_line = f.readline()
            # if "," in first_line or "," in second_line:
            #     delimiter = ","
            # else:
            #     # regular expression, indicating one or more whitespace
            #     delimiter = "\s+"
            return cls.from_csv(
                role=role,
                abs_path=abs_path,
                dataset_type=dataset_type,
                delimiter=delimiter,
                has_header=has_header,
                date_columns=date_columns,
                row_threshold=row_threshold,
                column_threshold=column_threshold,
                mappings=mappings,
                transform=transform,
            )

        elif extension in ("xls", "xlsx"):
            return cls.from_excel(
                role=role,
                abs_path=abs_path,
                dataset_type=dataset_type,
                has_header=has_header,
                date_columns=date_columns,
                row_threshold=row_threshold,
                column_threshold=column_threshold,
                mappings=mappings,
                transform=transform,
            )

        elif extension in ("json",):
            data_field = "data"
            return cls.from_json(
                role=role,
                abs_path=abs_path,
                dataset_type=dataset_type,
                data_field=data_field,
                date_columns=date_columns,
                row_threshold=row_threshold,
                column_threshold=column_threshold,
                mappings=mappings,
                transform=transform,
            )

        else:
            raise RuntimeError("file type {} is not supported.".format(extension))

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value: list):
        assert len(value) == len(self._header), (
            "the length of the new header is {}, which does not match the length"
            "of the older header".format(len(value))
        )
        self._header = value

    @property
    def ids(self):  # read only
        """Always return a Python list"""
        # avoid re-computing on each function call
        if not hasattr(self, "_ids"):
            raw_ids = self._raw_dataset[:, 0]
            # the data type of _ids must be native Python integers, so it can be
            # compatible with the Private Set Intersection(PSI) module
            py_ids = [int(_id.item()) for _id in raw_ids]
            setattr(self, "_ids", py_ids)
        return getattr(self, "_ids")

    def obfuscated_ids(self, option="md5"):
        import hashlib

        from gmssl import func, sm3

        assert option in ("md5", "sha256", "sm3"), (
            "ids obfuscation option can only take from md5, sha256, sm3, but {} got."
            .format(option)
        )

        raw_ids = self.ids
        obfuscated_ids = []
        if option in ("md5", "sha256"):
            if option == "md5":
                obfuscate_func = hashlib.md5
            else:
                obfuscate_func = hashlib.sha256
            for _id in raw_ids:
                _id_encode = str(_id).encode()
                _id_hash = obfuscate_func(_id_encode).hexdigest()
                _id_int = int(_id_hash, 16)
                obfuscated_ids.append(_id_int)
        else:
            for _id in raw_ids:
                _id_bytes = bytes(str(_id), encoding="utf-8")
                _id_hash = sm3.sm3_hash(func.bytes_to_list(_id_bytes))
                _id_int = int(_id_hash, 16)
                obfuscated_ids.append(_id_int)

        return obfuscated_ids

    @property
    def features(self):  # read only
        if not hasattr(self, "_features"):
            if self.role == Const.ACTIVE_NAME:
                setattr(self, "_features", self._raw_dataset[:, 2:])
            else:
                setattr(self, "_features", self._raw_dataset[:, 1:])
        return getattr(self, "_features")

    @property
    def labels(self):  # read only
        if self.role == Const.PASSIVE_NAME:
            raise AttributeError("Passive party has no labels.")

        if not hasattr(self, "_labels"):
            raw_labels = self._raw_dataset[:, 1]
            if self.dataset_type == Const.REGRESSION:  # regression dataset
                setattr(self, "_labels", raw_labels)

            else:  # classification dataset, neet to convert label values to integers
                if isinstance(raw_labels, np.ndarray):
                    raw_labels = raw_labels.astype(np.int32)  # NumPy
                else:
                    # the dtype of _labels should be cast to torch.long, otherwise,
                    # "the RuntimeError: expected scalar type Long but found Int"
                    # will be raised
                    raw_labels = raw_labels.type(torch.long)  # PyTorch
                setattr(self, "_labels", raw_labels)
        return getattr(self, "_labels")

    @property
    def n_features(self):  # read only
        return self.features.shape[1]

    @property
    def n_samples(self):  # read only
        return self.features.shape[0]

    def filter(self, intersect_ids: Union[list, np.ndarray]):
        # Solution 1: this works only when dataset ids start from zero
        # if type(intersect_ids) == list:
        #     intersect_ids = np.array(intersect_ids)
        # self.np_dataset = self.np_dataset[intersect_ids]

        # Solution 2: this works only when dataset ids are sorted and ascending
        # reference: https://stackoverflow.com/a/12122989/8418540
        # if type(intersect_ids) == list:
        #     intersect_ids = np.array(intersect_ids)
        # all_ids = np.array(self.ids)
        # idxes = np.searchsorted(all_ids, intersect_ids)
        # self.np_dataset = self.np_dataset[idxes]

        # Solution 3: more robust, but slower
        if isinstance(intersect_ids, list):
            pass
        elif isinstance(intersect_ids, np.ndarray):
            intersect_ids = intersect_ids.tolist()
        else:
            raise TypeError(
                "intersect_ids dtype is expected to be list or np.ndarray,"
                "but got {}".format(type(intersect_ids))
            )

        idxes = []
        all_ids = np.array(self.ids)
        for id_value in intersect_ids:
            idx = np.where(all_ids == id_value)[0][0]
            idxes.append(idx)
        new_raw_dataset = self._raw_dataset[idxes]
        self.set_dataset(new_raw_dataset)

    def filter_fields(self, target_fields, excluding_fields=False):
        offset = 1 if self.role == Const.PASSIVE_NAME else 2
        feats_header = self._header[offset:]
        all_idxes = list(range(len(feats_header)))
        if isinstance(target_fields[0], str):
            selected_idxes = []
            for field in target_fields:
                idx = feats_header.index(field)
                selected_idxes.append(idx)
            if excluding_fields:
                selected_idxes = list(set(all_idxes) - set(selected_idxes))
        elif isinstance(target_fields[0], int):
            selected_idxes = target_fields
            if excluding_fields:
                selected_idxes = list(set(all_idxes) - set(selected_idxes))
        else:
            raise TypeError(
                "each element in target_fields should be a str or an int, "
                "but got {} instead.".format(type(target_fields[0]))
            )

        if self.role == Const.PASSIVE_NAME:
            column_idxes = [0] + (np.array(selected_idxes) + offset).tolist()
        else:
            column_idxes = [0, 1] + (np.array(selected_idxes) + offset).tolist()

        new_raw_dataset = self._raw_dataset[:, column_idxes]
        new_header = (np.array(self._header)[column_idxes]).tolist()
        self.set_dataset(new_raw_dataset)
        self._header = new_header

    def describe(self, path="./"):
        import io
        import os

        import seaborn as sns
        from matplotlib import pyplot as plt
        from termcolor import colored

        from linkefl.feature.feature_evaluation import FeatureEvaluation
        from linkefl.feature.woe import Basewoe

        static_result = {}
        static_result["n_samples"] = self.n_samples
        static_result["n_features"] = self.n_features

        pd.set_option("display.max_columns", None)
        df_dataset = pd.DataFrame(self.features)
        # for i in range(self.n_features):
        #     df_dataset.rename(columns={i: 'x{}'.format(i)}, inplace=True)
        if self.role == Const.ACTIVE_NAME:
            for i in range(self.n_features):
                df_dataset.rename(columns={i: self.header[i + 2]}, inplace=True)
        else:
            for i in range(self.n_features):
                df_dataset.rename(columns={i: self.header[i + 1]}, inplace=True)

        # Calculate the unique value.
        col_names = list(df_dataset.columns.values)
        num_unique_data = np.array(df_dataset[col_names].nunique().values)
        num_unique = pd.DataFrame(
            data=num_unique_data.reshape((1, -1)), index=["unique"], columns=col_names
        )

        # Calculate the top value.
        col_sum = df_dataset.sum().values.reshape((1, -1))
        col_top3 = np.array([])
        for col in col_names:
            temp = df_dataset.nlargest(3, col)[col].values.reshape((-1, 1))
            col_top3 = (
                temp if col_top3.size == 0 else np.concatenate((col_top3, temp), axis=1)
            )
        top3_ratio_data = col_top3 / col_sum
        top3_ratio = pd.DataFrame(
            data=top3_ratio_data, index=["top1", "top2", "top3"], columns=col_names
        )

        if self.role == Const.ACTIVE_NAME:
            # Calculate the iv value and iv_rate.
            iv_idxes = list(range(self.n_features))
            _, _, iv = Basewoe(dataset=self, idxes=iv_idxes)._cal_woe(
                self.labels, "active", modify=False
            )
            iv = pd.DataFrame(iv, index=[0])
            iv = pd.DataFrame(data=iv.values, index=["iv"], columns=col_names)
            iv_sum = iv.iloc[0, :].sum()
            iv_rate = pd.DataFrame(
                data=iv.values / iv_sum, index=["iv_rate"], columns=col_names
            )
            # Calculate the xgb_importance.
            importance, _ = FeatureEvaluation.tree_importance(self, save_pic=False)
            importance = importance.reshape(1, -1)
            importance = pd.DataFrame(
                data=importance, index=["xgb_importance"], columns=col_names
            )
        else:
            iv = pd.DataFrame(
                data=np.zeros((1, self.n_features)), index=["iv"], columns=col_names
            )
            iv_rate = pd.DataFrame(
                data=np.zeros((1, self.n_features)),
                index=["iv_rate"],
                columns=col_names,
            )
            importance = pd.DataFrame(
                data=np.zeros((1, self.n_features)),
                index=["xgb_importance"],
                columns=col_names,
            )

        info = pd.concat(
            [df_dataset.describe(), num_unique, top3_ratio, iv, iv_rate, importance]
        )
        info = info.round(4)

        stat = {}
        for field in col_names:
            tstat = {}
            tstat["missing_rate"] = round(
                ((self.n_samples - info.loc["count"][field]) / self.n_samples), 4
            )
            tstat["range"] = "[{}, {}]".format(
                info.loc["min"][field], info.loc["max"][field]
            )
            tstat["unique"] = int(info.loc["unique"][field])
            tstat["iv"] = float(info.loc["iv"][field])
            tstat["iv_rate"] = float(info.loc["iv_rate"][field])
            tstat["xgb_importance"] = float(info.loc["xgb_importance"][field])
            # tstat['top'] = float(info.loc['top1'][field])
            tstat["top"] = random.random()
            tstat["mean"] = float(info.loc["mean"][field])
            tstat["quartile"] = float(info.loc["25%"][field])
            tstat["max"] = float(info.loc["max"][field])
            tstat["min"] = float(info.loc["min"][field])
            tstat["std"] = float(info.loc["std"][field])
            tstat["median"] = float(info.loc["50%"][field])
            stat[field] = tstat
        static_result["stat"] = stat

        # Plot max/min/median pictures.
        CommonDataset._plot_bar(
            col_names, info.loc["min", :].values, "Min Value", "min_plot", path
        )
        CommonDataset._plot_bar(
            col_names, info.loc["max", :].values, "Max Value", "max_plot", path
        )
        CommonDataset._plot_bar(
            col_names, info.loc["50%", :].values, "Median Value", "mid_plot", path
        )
        CommonDataset._plot_bar(
            col_names, info.loc["std", :].values, "std value", "std_plot", path
        )
        CommonDataset._plot_box(
            self.features[:, :10], col_names[:10], path
        )

        return static_result

    def get_dataset(self):
        return self._raw_dataset

    def set_dataset(self, new_raw_dataset: Union[np.ndarray, torch.Tensor]):
        # must delete old properties to save memory
        if hasattr(self, "_raw_dataset"):
            del self._raw_dataset
        if hasattr(self, "_ids"):
            del self._ids
        if hasattr(self, "_features"):
            del self._features
        if hasattr(self, "_labels"):
            del self._labels

        # update new property
        self._raw_dataset = new_raw_dataset

    @staticmethod
    def _load_buildin_dataset(
        role, name, root, train, download, frac, perm_option, seed=None
    ):
        import os
        from urllib.error import URLError

        from linkefl.feature import cal_importance_ranking
        from linkefl.util import urlretrive

        def _check_exists(dataset_name, root_, train_, resources_):
            if train_:
                filename_ = resources_[dataset_name][0]
            else:
                filename_ = resources_[dataset_name][1]
            return os.path.exists(os.path.join(root_, filename_))

        resources = {
            "cancer": ("cancer-train.csv", "cancer-test.csv"),
            "digits": ("digits-train.csv", "digits-test.csv"),
            "diabetes": ("diabetes-train.csv", "diabetes-test.csv"),
            "iris": ("iris-train.csv", "iris-test.csv"),
            "wine": ("wine-train.csv", "wine-test.csv"),
            "epsilon": ("epsilon-train.csv", "epsilon-test.csv"),
            "census": ("census-train.csv", "census-test.csv"),
            "credit": ("credit-train.csv", "credit-test.csv"),
            "default_credit": ("default-credit-train.csv", "default-credit-test.csv"),
            "covertype": ("covertype-train.csv", "covertype-test.csv"),
            "criteo": ("criteo-train.csv", "criteo-test.csv"),
            "higgs": ("higgs-train.csv", "higgs-test.csv"),
            "year": ("year-train.csv", "year-test.csv"),
            "nyc_taxi": ("nyc-taxi-train.csv", "nyc-taxi-test.csv"),
            "avazu": ("avazu-train.csv", "avazu-test.csv"),
        }
        BASE_URL = "http://47.96.163.59:80/datasets/"
        root = os.path.join(root, "tabular")

        if download:
            if _check_exists(name, root, train, resources):
                # if data files have already been downloaded, then skip this branch
                print("Data files have already been downloaded.")
            else:
                # download data files from web server
                os.makedirs(root, exist_ok=True)
                filename = resources[name][0] if train else resources[name][1]
                fpath = os.path.join(root, filename)
                full_url = BASE_URL + filename
                try:
                    print("Downloading {} to {}".format(full_url, fpath))
                    urlretrive(full_url, fpath)
                except URLError as error:
                    raise RuntimeError(
                        "Failed to download {} with error message: {}".format(
                            full_url, error
                        )
                    )
                print("Done!")
        if not _check_exists(name, root, train, resources):
            raise RuntimeError(
                "Dataset not found. You can use download=True to get it."
            )

        # ===== 1. Load dataset =====
        if train:
            np_csv = np.genfromtxt(
                os.path.join(root, resources[name][0]), delimiter=",", encoding="utf-8"
            )
        else:
            np_csv = np.genfromtxt(
                os.path.join(root, resources[name][1]), delimiter=",", encoding="utf-8"
            )
        _ids = np_csv[:, 0]  # no need to convert to integers here
        _labels = np_csv[:, 1]  # no need to convert to integers here
        _feats = np_csv[:, 2:]
        _feats_header = ["x{}".format(i) for i in range(_feats.shape[1])]

        # ===== 2. Apply feature permutation =====
        if perm_option == Const.SEQUENCE:
            permuted_feats = _feats
            permuted_header = _feats_header
        elif perm_option == Const.RANDOM:
            if seed is not None:
                random.seed(seed)
            perm = list(range(_feats.shape[1]))
            random.shuffle(perm)
            permuted_feats = _feats[:, perm]
            permuted_header = np.array(_feats_header)[perm].tolist()
            del _feats  # save memory
        elif perm_option == Const.IMPORTANCE:
            rankings = cal_importance_ranking(name, _feats, _labels)
            permuted_feats = _feats[:, rankings]
            permuted_header = np.array(_feats_header)[rankings].tolist()
        else:
            raise ValueError("Invalid permutation option.")

        # ===== 3. Split feature =====
        num_passive_feats = int(frac * permuted_feats.shape[1])
        if role == Const.PASSIVE_NAME:
            splitted_feats = permuted_feats[:, :num_passive_feats]
            header = ["id"] + permuted_header[:num_passive_feats]
            np_dataset = np.concatenate((_ids[:, np.newaxis], splitted_feats), axis=1)
        else:
            splitted_feats = permuted_feats[:, num_passive_feats:]
            header = ["id"] + ["y"] + permuted_header[num_passive_feats:]
            np_dataset = np.concatenate(
                (_ids[:, np.newaxis], _labels[:, np.newaxis], splitted_feats), axis=1
            )

        return np_dataset, header

    @staticmethod
    def _date_data(
        df_dataset: pd.DataFrame,
        columns = None,
    ):
        if columns is None:
            return df_dataset
        for column in columns:
            df_dataset[column] = pd.to_datetime(df_dataset[column])
        return df_dataset

    @staticmethod
    def _clean_data(
        df_dataset: pd.DataFrame,
        row_threshold: float = 0.3,
        column_threshold: float = 0.3,
    ):
        """Remove rows and columns with too many NANs

        Parameters
        ----------
        df_dataset : pd.DataFrame
            dataset which needs clean
        row_threshold : float
            in a row, the threshold of NANs
        column_threshold : float
            in a column, the threshold of NANs

        Returns
        -------
        pd.DataFrame
            a cleaned df_dataset (with acceptable NANs)
        """

        df_nan = df_dataset.isna()
        rows, columns = df_dataset.shape
        # check row
        row_indexes = df_nan.sum(axis=1) < columns * row_threshold
        # check column
        column_indexes = df_nan.sum(axis=0) < rows * column_threshold
        new_df_dataset = df_dataset.loc[row_indexes, column_indexes]

        return new_df_dataset

    @staticmethod
    def _outlier_data(df_dataset: pd.DataFrame, role):
        from pandas.core.dtypes.common import is_numeric_dtype

        # start = 2 if role == Const.ACTIVE_NAME else 1
        start = 2
        for i in range(start, df_dataset.shape[1]):
            column_data = df_dataset.iloc[:, i]
            if not is_numeric_dtype(column_data):
                continue
            column_data_mean = np.mean(column_data)
            column_data_std = np.std(column_data)
            outliers = np.abs(column_data - column_data_mean) > 1 * column_data_std
            df_dataset.loc[outliers, df_dataset.columns[i]] = pd.NA
        return df_dataset

    @staticmethod
    def _fill_data(df_dataset: pd.DataFrame):
        new_df_dataset = df_dataset.apply(lambda x: x.fillna(x.value_counts().index[0]))
        return new_df_dataset

    @staticmethod
    def _pandas2numpy(df_dataset: pd.DataFrame, mappings: dict = None):
        """Transform a pandas DataFrame into Numpy Array"""
        from pandas.core.dtypes.common import is_numeric_dtype

        if mappings is None:
            create_mappings_flag = True
            mappings = dict()
        else:
            create_mappings_flag = False
            mappings = mappings

        for i, dtype in enumerate(df_dataset.dtypes):
            if not is_numeric_dtype(dtype):
                if create_mappings_flag is True:
                    df_dataset[df_dataset.columns[i]], uniques = pd.factorize(
                        df_dataset.iloc[:, i]
                    )
                    mapping = dict(zip(uniques, range(len(uniques))))
                    mappings[i] = mapping
                else:
                    df_dataset[df_dataset.columns[i]] = df_dataset.iloc[:, i].replace(
                        mappings[i]
                    )

        CommonDataset.mappings = mappings

        np_dataset = df_dataset.to_numpy()
        return np_dataset

    @staticmethod
    def _get_selected_fields(
        db_type, cursor, table, target_fields, excluding_fields, conn=None
    ):
        if db_type == "db2":
            import ibm_db

            sql = "SELECT * FROM {}".format(table)
            stmt = ibm_db.exec_immediate(conn, sql)
            result = ibm_db.fetch_both(stmt)
            # result = {'ID':1, 0:1, 'USER_NAME':'xxx', 1:'xxx'}
            keys = list(result.keys())
            all_fields = keys[::2]
            # all_fields = ['ID', 'USER_NAME']
        else:
            if db_type == "oracle":
                sql = "select * from {} fetch first 1 rows only".format(table)
            else:
                sql = "select * from {} limit 1".format(table)
            cursor.execute(sql)
            # description is a tuple of tuple,
            # the first position of tuple element is the field name
            all_fields = [tuple_[0] for tuple_ in cursor.description]

        if target_fields is None:
            selected_fields = all_fields
        else:
            if not excluding_fields:
                selected_fields = target_fields
            else:
                selected_fields = list(set(all_fields) - set(target_fields))

        return selected_fields

    @staticmethod
    def _gen_header(role, n_feats):
        feats_header = ["x{}".format(i) for i in range(n_feats)]
        if role == Const.ACTIVE_NAME:
            header = ["id"] + ["y"] + feats_header
        else:
            header = ["id"] + feats_header

        return header

    @staticmethod
    def _plot_bar(x, y, ylabel, file_name, path):
        import os

        from matplotlib import pyplot as plt

        if len(x) > 10:
            x = x[:10]
            y = y[:10]
        plt.bar(x, y, color="dodgerblue")
        plt.xlabel("Feature")
        plt.ylabel(ylabel)
        plt.title("{} of Each Feature".format(ylabel))
        plt.savefig(os.path.join(path, "{}.png".format(file_name)), pad_inches="tight")
        plt.close()

    @staticmethod
    def _plot_box(data, labels, path):
        import os

        from matplotlib import pyplot as plt

        plt.boxplot(data, labels=labels, showmeans=True, meanline=True)
        plt.title("Boxplot of features {}".format(labels))
        plt.savefig(os.path.join(path, 'box_plot.png'), pad_inches="tight")
        plt.close()


# if __name__ == "__main__":
    # from linkefl.feature.transform import OneHot
    #
    # print("the first df_dataset")
    # _df_dataset = pd.DataFrame(
    #     {"id": [0, 1, 2],
    #      "x": [1.1, 1.2, 1.3],
    #      "a": ["aaa", "bbb", "ccc"],
    #      "b": ["aa", "bb", "cc"]}
    # )
    # print(_df_dataset)
    # _np_dataset = CommonDataset._pandas2numpy(_df_dataset)
    # _mappings = CommonDataset.mappings
    # # you can save these mappings and load it back
    # # when loading testset at inference pahse
    # # with open('train_mappings.pkl', 'wb') as f:
    # #     pickle.dump(mappings, f)
    # print(_np_dataset)
    # _np_dataset = OneHot([1, 2]).fit(_np_dataset, Const.PASSIVE_NAME)
    # print(_np_dataset)
    #
    # print()
    #
    # print("the second df_dataset")
    # another_df_dataset = pd.DataFrame(
    #     {"id": [0, 1, 2],
    #      "x": [1.1, 1.2, 1.3],
    #      "a": ["bbb", "ccc", "aaa"],
    #      "b": ["cc", "aa", "bb"]}
    # )
    # print(another_df_dataset)
    # # you can load the mappings back and apply it to testset
    # # with open('train_mappings.pkl', 'rb') as f:
    # #     mappings = pickle.load(f)
    # another_np_dataset = CommonDataset._pandas2numpy(
    #     another_df_dataset,
    #     mappings=_mappings
    # )
    # print(another_np_dataset)
    # another_np_dataset = OneHot([1, 2]).fit(another_np_dataset, Const.PASSIVE_NAME)
    # print(another_np_dataset)

    # df_dataset_ = pd.DataFrame(
    #     {
    #         "id": [1, 2, 3, 4, 5],
    #         "x": [1.1, 1.2, np.nan, np.nan, 1.2],
    #         "a": ["a", "aa", "aaa", np.nan, "aaaaa"],
    #         "b": ["b", "bb", "bbb", np.nan, np.nan],
    #         "c": [np.nan, np.nan, np.nan, "cccc", "ccccc"]
    #     }
    # )
    # print("Original")
    # print(df_dataset_)
    # cleaned_df_dataset = CommonDataset._clean_data(
    #     df_dataset_,
    #     row_threshold=0.5,
    #     column_threshold=0.5
    # )
    # print("Cleaned")
    # print(cleaned_df_dataset)
    # filled_df_dataset = CommonDataset._fill_data(cleaned_df_dataset)
    # print("Filled")
    # print(filled_df_dataset)

    # df_dataset_ = pd.DataFrame(
    #     {
    #         "id": [1, 2, 3, 4, 5],
    #         "x": [1, 1, 1, 1, 10000],
    #         "a": [1, 2, 3, 4, "5"],
    #     }
    # )
    # print("Original")
    # print(df_dataset_)
    # new_df_dataset = CommonDataset._outlier_data(df_dataset_, role=Const.PASSIVE_NAME)
    # print("New")
    # print(new_df_dataset)

    # abs_path = "/Users/tanjuntao/LinkeFL-Servicer/data/电商平台精准营销数据202206.csv"
    # np_dataset = CommonDataset.from_csv(
    #     role=Const.ACTIVE_NAME,
    #     abs_path=abs_path,
    #     dataset_type=Const.CLASSIFICATION,
    # )
    # print(np_dataset.header)
    # print(np_dataset.header_type)
