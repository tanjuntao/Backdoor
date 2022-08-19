import numpy as np


# def _greedy_find_bin(vec, max_bin):
#     """传入 feature，返回每个值所在 bin 编号和划分点
#
#     Args:
#         vec: 一维 numpy 数组，横竖皆可
#
#     Returns:
#         vec_bin_index: 一维 numpy 数组，与 vec 形状相同，为 vec 中每个值对应的 bin 编号
#         vec_split: 一维 list 数组，为相邻 bin 的划分点
#     """
#
#     # 记录原始形状
#     shape = vec.shape
#     vec = vec.flatten()
#
#     vec_bin_index = np.empty_like(vec, dtype=int)  # vec 中每个值对应的 bin 编号
#     vec_bin_start = []  # bin 起始
#     vec_bin_end = []  # bin 终止
#     vec_split = []  # 两 bin 切分点
#
#     min_data_in_bin = SBTConfig.MIN_BIN_SAMPLES
#
#     distinct_vec, counts = np.unique(vec, return_counts=True)
#
#     if len(distinct_vec) <= max_bin:
#         # 如果唯一的特征值的数目比 max_bin 还小
#
#         cur_cnt_bin = 0
#         index = 0
#
#         for i, data in enumerate(distinct_vec):
#             if cur_cnt_bin == 0:
#                 # 一个新的 bin 起始
#                 vec_bin_start.append(data)
#
#             vec_bin_index[vec == data] = index
#             cur_cnt_bin += counts[i]
#
#             # 满足如下任意一个条件的时候 bin 终止
#             # 1. 当前数量已达到 min_data_in_bin
#             # 2. 当前为最后一个特征
#             if cur_cnt_bin >= min_data_in_bin or i == len(distinct_vec) - 1:
#                 vec_bin_end.append(data)
#                 cur_cnt_bin = 0
#                 index += 1
#
#     else:
#         # 唯一的特征值的数目比 max_bin 大，出现多个特征值公用一个 bin 的情况
#
#         if min_data_in_bin > 0:
#             # 判断 max_bin 的大小
#             max_bin = min(max_bin, len(vec) // min_data_in_bin)
#             max_bin = max(max_bin, 1)
#
#         # 一个 bin 的大小--动态更新的
#         mean_bin_size = len(vec) / max_bin
#
#         rest_bin_cnt = max_bin
#         rest_sample_cnt = len(vec)
#         is_big_count_value = [False] * len(distinct_vec)  # 标记数组
#
#         # 如果一个特征取值数超过 mean_bin_size，则需要单独一个 bin
#         for i, count in enumerate(counts):
#             if count > mean_bin_size:
#                 is_big_count_value[i] = True
#                 rest_bin_cnt -= 1
#                 rest_sample_cnt -= count
#
#         # 剩下的特征取值中平均每个 bin 的取值个数
#         mean_bin_size = rest_sample_cnt / rest_bin_cnt
#
#         cur_cnt_bin = 0
#         index = 0
#
#         for i, data in enumerate(distinct_vec):
#             if cur_cnt_bin == 0:
#                 # 一个新的 bin 起始
#                 vec_bin_start.append(data)
#
#             if not is_big_count_value[i]:
#                 # 标记为 True 的时候不用减，因为之前已经减过了
#                 rest_sample_cnt -= counts[i]
#
#             vec_bin_index[vec == data] = index
#             cur_cnt_bin += counts[i]
#
#             # 满足如下任意一个条件的时候 bin 终止
#             # 1. 当前特征需要单独成 bin
#             # 2. 当前数量已达到 mean_bin_size
#             # 3. 当前为最后一个特征
#             # 4. 下一个特征要单独成 bin，且当前已满足一定条件
#             if (
#                 is_big_count_value[i]
#                 or cur_cnt_bin >= mean_bin_size
#                 or i == len(distinct_vec) - 1
#                 or (is_big_count_value[i + 1] and cur_cnt_bin >= max(1, mean_bin_size / 2))
#             ):
#                 vec_bin_end.append(data)
#                 cur_cnt_bin = 0
#                 index += 1
#
#                 if not is_big_count_value[i]:
#                     # 当前是基于小数的 bin，需更新 mean_bin_size
#                     rest_bin_cnt -= 1
#                     if rest_bin_cnt == 0:
#                         # 所有小数应当已经划分完毕
#                         assert rest_sample_cnt == 0
#                     else:
#                         mean_bin_size = rest_sample_cnt / rest_bin_cnt
#
#     # # 对每个 bin 计算划分点，划分点 = (当前 bin 的最大特征值 + 下一个 bin 的最小特征值) / 2
#     for i, end in enumerate(vec_bin_end[:-1]):
#         vec_split.append((end + vec_bin_start[i + 1]) / 2)
#
#     # 还原为原始形状
#     vec_bin_index = vec_bin_index.reshape(shape)
#
#     return vec_bin_index, vec_split


def _find_bin(vec, max_bin):
    shape = vec.shape
    vec = vec.flatten()


    distinct_vec, counts = np.unique(vec, return_counts=True)
    sorted_vec = np.sort(vec)

    bin_data_rate = 1 / max_bin
    bin_last_pos = [sorted_vec[int((i + 1) * bin_data_rate * len(vec)) - 1] for i in range(max_bin)]
    vec_bin_end = np.unique(bin_last_pos)

    vec_bin_index = np.empty_like(vec, dtype=int)
    index = 0
    for data, count in zip(distinct_vec, counts):
        if data > vec_bin_end[index]:
            index += 1
        vec_bin_index[vec == data] = index
    vec_bin_index = vec_bin_index.reshape(shape)

    return vec_bin_index, vec_bin_end[:-1]


def get_bin_info(x_train, max_bin):
    """compute hist information

    Args:
        x_train: training data，size = sample * feature
        max_bin: max bin number for a feature point

    Returns:
        bin_index: bin index of each feature point in the complete feature hist (a column)，size = sample * feature
        bin_split: split point in the complete feature hist (a column)，size = feature * bin_of_this_feature
    """

    bin_index = np.empty_like(x_train, dtype=int)
    bin_split = [None] * x_train.shape[1]
    for i in range(x_train.shape[1]):
        # find hist for each feature point
        bin_index[:, i], bin_split[i] = _find_bin(x_train[:, i], max_bin)

    # fill the empty place in bin_split to solve the influence of different bin_num (seems no influence right now)
    # bin_split = np.array(list(itertools.zip_longest(*bin_split, fillvalue=None))).T

    return bin_index, bin_split


def wrap_message(name, *, content):
    return {"name": name, "content": content}
