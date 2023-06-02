import numpy as np


# todo: check reg_gamma here.
def find_split(hist_list, task, reg_lambda, reg_gamma):
    """find the best split point in a list of hists

    :returns:
        best split point -> (hist_id, feature_id, split_id, gain)
    """
    max_hist_id = None
    max_feature_id = None
    max_split_id = None
    max_gain = -1

    for hist_id, hist in enumerate(hist_list):
        # hist.bin_gh:
        # 3d array in binary，featureNum * binNum * 2
        # 4d array in multi，featureNum * binNum * config["train"]["classNum"] * 2
        if hist is None:
            continue

        for feature_id, feature_bin_gh in enumerate(hist.bin_gh):
            for split_id in range(len(feature_bin_gh) - 1):
                left_bin_gh, right_bin_gh = np.split(feature_bin_gh, [split_id + 1])

                if task == "binary" or task == "regression":
                    gain = _split_gain(
                        feature_bin_gh, left_bin_gh, right_bin_gh, reg_lambda, reg_gamma
                    )
                elif task == "multi":
                    gain = _split_gain_multi(
                        feature_bin_gh, left_bin_gh, right_bin_gh, reg_lambda, reg_gamma
                    )
                else:
                    raise ValueError("No such task label.")

                if max_gain is None or max_gain < gain:
                    max_hist_id = hist_id
                    max_feature_id = feature_id
                    max_split_id = split_id
                    max_gain = gain

    return max_hist_id, max_feature_id, max_split_id, max_gain


def _split_gain(bin_gh, bin_gh_left, bin_gh_right, reg_lambda, reg_gamma):
    """Calculates gain in the given situation"""
    left_score = _structure_score(bin_gh_left, reg_lambda)
    right_score = _structure_score(bin_gh_right, reg_lambda)
    current_score = _structure_score(bin_gh, reg_lambda)

    gain = 0.5 * (left_score + right_score - current_score) - reg_gamma

    return gain


def _split_gain_multi(bin_gh, bin_gh_left, bin_gh_right, reg_lambda, reg_gamma):
    left_score = _structure_score_multi(bin_gh_left, reg_lambda)
    right_score = _structure_score_multi(bin_gh_right, reg_lambda)
    current_score = _structure_score_multi(bin_gh, reg_lambda)
    # todo: check new cal func here.
    gain = 0.5 * (left_score + right_score - current_score) - reg_gamma

    return gain


def _structure_score(bin_gh, reg_lambda):
    gradient = bin_gh[:, :1]
    hessian = bin_gh[:, 1:2]

    g_2 = np.power(gradient.sum(), 2)
    h = hessian.sum()

    return 0.5 * (g_2 / (h + reg_lambda))


def _structure_score_multi(bin_gh, reg_lambda):
    bin_g = bin_gh[:, :, 0]
    bin_h = bin_gh[:, :, 1]

    score = 0
    for label_id in range(bin_g.shape[1]):
        g_id = bin_g[:, label_id].flatten()
        h_id = bin_h[:, label_id].flatten()
        score += np.power(g_id, 2).sum() / (h_id.sum() + reg_lambda)

    return score
