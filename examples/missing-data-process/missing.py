import numpy as np
import pandas as pd

from linkefl.common.const import Const
from linkefl.dataio.common_dataset import CommonDataset

df_dataset_ = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "x": [1.1, 1.2, np.nan, np.nan, 1.2],
        "a": ["a", "aa", "aaa", np.nan, "aaaaa"],
        "b": ["b", "bb", "bbb", np.nan, np.nan],
        "c": [np.nan, np.nan, np.nan, "cccc", "ccccc"],
        "d": [1, 1, 1, 1, 10000000],
        "e": ["2022/3/15", np.nan, "12/24/2021", "15-03-2022", "2023-02-20"]
    }
)
print("Original")
print(df_dataset_)

# 异常日期处理
dated_df_dataset = CommonDataset._date_data(df_dataset_, columns=["e"])
print("Dated")
print(dated_df_dataset)

# 行列缺失比例删除
cleaned_df_dataset = CommonDataset._clean_data(dated_df_dataset, row_threshold=0.5, column_threshold=0.5)
print("Cleaned")
print(cleaned_df_dataset)

# 异常字段清除
unout_df_dataset = CommonDataset._outlier_data(cleaned_df_dataset, role=Const.PASSIVE_NAME)
print("Unout")
print(unout_df_dataset)

# 空值填充
filled_df_dataset = CommonDataset._fill_data(unout_df_dataset)
print("Filled")
print(filled_df_dataset)
