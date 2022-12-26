import numpy as np


# def goss_sampling(grad, hess):
#     """
#     GOSS 单边采样，基于light GBM的思想；
#     return select_grad, select_hess, select_idx.
#     """
#     # if it is multi-classification case, we need to sum g
#     g_sum_arr = np.abs(grad).sum(axis=1)
#
#     # abs_g_list_arr = g_sum_arr
#     sorted_idx = np.argsort(-g_sum_arr, kind="stable")  # stable sample result
#
#     sample_num = len(g_sum_arr)
#     a_part_num = int(sample_num * SBTConfig.TOP_RATE)
#     b_part_num = int(sample_num * SBTConfig.OTHER_RATE)
#
#     if a_part_num == 0 or b_part_num == 0:
#         raise ValueError("subsampled result is 0: top sample {}, other sample {}".format(a_part_num, b_part_num))
#
#     # index of a part
#     a_sample_idx = sorted_idx[:a_part_num]
#
#     # index of b part
#     rest_sample_idx = sorted_idx[a_part_num:]
#     b_sample_idx = np.random.choice(rest_sample_idx, size=b_part_num, replace=False)
#
#     # small gradient sample weights
#     amplify_weights = (1 - SBTConfig.TOP_RATE) / SBTConfig.OTHER_RATE
#     grad[b_sample_idx] *= amplify_weights
#     hess[b_sample_idx] *= amplify_weights
#
#     # get selected sample
#     a_idx_set, b_idx_set = set(list(a_sample_idx)), set(list(b_sample_idx))
#     idx_set = a_idx_set.union(b_idx_set)
#     selected_idx = np.array(list(idx_set))
#
#     selected_g, selected_h = grad[selected_idx], hess[selected_idx]
#
#     return selected_g, selected_h, selected_idx


def gh_packing(grad, hess, r):
    assert r < 64

    g_int_max = round(len(grad) * np.abs(grad).max() * (1 << r))
    h_int_max = round(len(hess) * np.abs(hess).max() * (1 << r))

    g_length = g_int_max.bit_length()
    h_length = h_int_max.bit_length()
    gh_length = (g_length + 1) + h_length

    # transform g, h from float to big int
    grad_int = np.floor(grad * (1 << 64)) >> (64 - r)
    hess_int = np.floor(hess * (1 << 64)) >> (64 - r)

    # pack g, h
    gh_int = (grad_int << h_length) + hess_int

    return gh_int, h_length, gh_length


def leaf_weight(gh, sample_tag, reg_lambda):
    g_sum, h_sum = np.dot(sample_tag, gh)
    weight = -(g_sum / (h_sum + reg_lambda))

    return weight


def _structure_score(bin_gh, reg_lambda):
    gradient = bin_gh[:, :1]
    hessian = bin_gh[:, 1:2]

    g_2 = np.power(gradient.sum(), 2)
    h = hessian.sum()

    return 0.5 * (g_2 / (h + reg_lambda))


def _split_gain(bin_gh, bin_gh_left, bin_gh_right, reg_lambda):
    """Calculates gain in the given situation"""

    left_score = _structure_score(bin_gh_left, reg_lambda)
    right_score = _structure_score(bin_gh_right, reg_lambda)
    current_score = _structure_score(bin_gh, reg_lambda)

    gain = 0.5 * (left_score + right_score - current_score) - reg_lambda

    return gain


def gh_compress_multi(grad, hess, r, key_size):
    gh_int, h_length, gh_length = gh_packing(grad, hess, r)

    plaintext_bit = key_size - 1
    capacity = int(plaintext_bit / gh_length)  # max number of gh in a cipher space

    gh_compress = []

    for sample_id in range(grad.shape[0]):
        gh_compress_vec, big_num, count = [], 0, 0

        for label_id in range(grad.shape[1]):
            gh = gh_int[sample_id][label_id]
            big_num = (big_num << gh_length) + gh
            count += 1

            if count == capacity:
                gh_compress_vec.append(big_num)
                big_num, count = 0, 0

        if count != 0:
            gh_compress_vec.append(big_num)

        gh_compress.append(gh_compress_vec)

    return np.array(gh_compress), h_length, gh_length


def leaf_weight_multi(gh, sample_tag, reg_lambda):
    grad = gh[:, :, 0]
    hess = gh[:, :, 1]
    sample_id = [i for i, tag in enumerate(sample_tag) if tag == 1]
    g_sum = grad[sample_id].sum(axis=0)
    h_sum = hess[sample_id].sum(axis=0)

    weight = -(g_sum / h_sum + reg_lambda)

    return weight


def _score_multi(bin_gh, reg_lambda):
    bin_g = bin_gh[:, :, 0]
    bin_h = bin_gh[:, :, 1]

    total = 0
    for label_id in range(bin_g.shape[1]):
        g_id = bin_g[:, label_id].flatten()
        h_id = bin_h[:, label_id].flatten()
        total += np.power(g_id, 2).sum() / (h_id.sum() + reg_lambda)

    score = -0.5 * total

    return score


def _split_gain_multi(bin_gh, bin_gh_left, bin_gh_right, reg_lambda):
    left_score = _score_multi(bin_gh_left, reg_lambda)
    right_score = _score_multi(bin_gh_right, reg_lambda)
    current_score = _score_multi(bin_gh, reg_lambda)

    gain = current_score - (left_score + right_score)

    return gain


def find_split(hist_list, task, reg_lambda):
    """find the best split point in a list of hists

    :returns:
        best split point -> (hist_id, feature_id, split_id, gain)
    """
    max_hist_id = None
    max_feature_id = None
    max_split_id = None
    max_gain = None

    for hist_id, hist in enumerate(hist_list):
        # hist.bin_gh: 3d array in binary，featureNum * binNum * 2
        #              4d array in multi，featureNum * binNum * config["train"]["classNum"] * 2
        if hist is None:
            continue

        for feature_id, feature_bin_gh in enumerate(hist.bin_gh):
            for split_id in range(len(feature_bin_gh) - 1):
                left_bin_gh, right_bin_gh = np.split(feature_bin_gh, [split_id + 1])

                if task == "binary" or task == "regression":
                    gain = _split_gain(feature_bin_gh, left_bin_gh, right_bin_gh, reg_lambda)
                elif task == "multi":
                    gain = _split_gain_multi(feature_bin_gh, left_bin_gh, right_bin_gh, reg_lambda)
                else:
                    raise ValueError("No such task label.")

                if max_gain is None or max_gain < gain:
                    max_hist_id = hist_id
                    max_feature_id = feature_id
                    max_split_id = split_id
                    max_gain = gain

    return max_hist_id, max_feature_id, max_split_id, max_gain

