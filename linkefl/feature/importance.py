import numpy as np
import shap
from xgboost import XGBClassifier, XGBRegressor, plot_importance

from linkefl.common.const import Const
from linkefl.dataio import NumpyDataset


# def _get_dataset(name):
#     """Load np_dataset by name.
#
#     The returned np_dataset will all be type np.ndarray.
#
#     Args:
#         name[str]: Name of np_dataset.
#
#     Returns:
#         x_train[np.ndarray]: Training set of X.
#         x_test[np.ndarray]: Testing set of X.
#         y_train[np.ndarray]: Training set of Y.
#         y_test[np.ndarray]: Testing set of Y.
#     """
#     if name == 'breast_cancer':
#         cancer = load_breast_cancer()
#         x_train, x_test, y_train, y_test = train_test_split(cancer.data,
#                                                             cancer.target,
#                                                             test_size=0.2,
#                                                             random_state=0)
#
#     elif name == 'digits':
#         X, Y = load_digits(return_X_y=True)
#         odd_idxes, even_idxes = np.where(Y % 2 == 1)[0], np.where(Y % 2 == 0)[0]
#         Y[odd_idxes] = 1
#         Y[even_idxes] = 0
#         x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
#                                                             random_state=0)
#
#     elif name == 'census_income':
#         train_set = np.genfromtxt('./raw_data/census_income_train.csv',
#                                   delimiter=',')
#         test_set = np.genfromtxt('./raw_data/census_income_test.csv',
#                                  delimiter=',')
#         x_train, y_train = train_set[:, 2:], train_set[:, 1]
#         x_test, y_test = test_set[:, 2:], test_set[:, 1]
#         y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
#
#
#     elif name == 'give_me_some_credit':
#         train_set = np.genfromtxt('./raw_data/give_me_some_credit_train.csv',
#                                   delimiter=',')
#         test_set = np.genfromtxt('./raw_data/give_me_some_credit_test.csv',
#                                  delimiter=',')
#         x_train, y_train = train_set[:, 2:], train_set[:, 1]
#         x_test, y_test = test_set[:, 2:], test_set[:, 1]
#         y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
#         print(train_set.shape[0] + test_set.shape[0], train_set.shape[1])
#
#     elif name in ('mnist', 'fashion_mnist'):
#         def __dataset(train):
#             if name == 'mnist':
#                 torch_dataset = datasets.MNIST(root='raw_data/data',
#                                                train=train,
#                                                download=True,
#                                                transform=ToTensor())
#             else:
#                 torch_dataset = datasets.FashionMNIST(root='raw_data/data',
#                                                       train=train,
#                                                       download=True,
#                                                       transform=ToTensor())
#
#             X, Y = [], []
#             for image, label in torch_dataset:
#                 image_flat = image.view(-1)
#                 image_numpy = image_flat.numpy()
#                 X.append(image_numpy)
#                 Y.append(label)
#
#             X, Y = np.array(X), np.array(Y)
#
#             return X, Y
#
#         x_train, y_train = __dataset(train=True)
#         x_test, y_test = __dataset(train=False)
#
#     else:
#         raise NotImplementedError('{} Dataset not supported now.'.format(name))
#
#     return x_train, x_test, y_train, y_test


def feature_ranking(dataset_name, measurement='xgboost'):
    """Calculate feature importances of each np_dataset and rank it in desending order.

    Args:
        dataset_name: Name of the np_dataset.
        measurement: What method will be used to calculate feature importance.
            Only "xgboost" and "shap" are valid options. If "xgboost" is used,
            we will use `feature_importance_` attribute of xgboost model to
            compute feature importance. If "shap" is used, we will use third-party
            tool Shap[https://github.com/slundberg/shap] to do it.

    Returns:
        ranking: An index vector indicating the feature importance in a
            desending order.
    """
    if dataset_name in ('cancer',
                        'digits',
                        'epsilon',
                        'census',
                        'credit',
                        'default_credit',
                        'diabetes',
                        'mnist',
                        'fashion_mnist',
                        ):
        ranking = permutation(dataset_name, measurement)
        return ranking

    # x_train, x_test, y_train, y_test = _get_dataset(dataset_name)
    passive_feat_frac = 0
    feat_perm_option = Const.SEQUENCE
    active_trainset = NumpyDataset.buildin_dataset(role=Const.ACTIVE_NAME,
                                                   dataset_name=dataset_name,
                                                   train=True,
                                                   passive_feat_frac=passive_feat_frac,
                                                   feat_perm_option=feat_perm_option)
    # simply treat dataset where the label column contains more than
    # 100 unique values as regression dataset
    if len(np.unique(active_trainset.labels)) > 100:
        model = XGBRegressor()
        model.fit(active_trainset.features, active_trainset.labels)
    else: # classification dataset
        if len(np.unique(active_trainset.labels)) == 2:
            # binary case: silent warnings of XGB Constructor
            model = XGBClassifier(use_label_encoder=False)
            model.fit(active_trainset.features, active_trainset.labels, eval_metric='logloss')
        else:
            model = XGBClassifier()
            model.fit(active_trainset.features, active_trainset.labels)

    if measurement == 'shap':
        explainer = shap.Explainer(model)
        # type(shap_values) = <class 'shap._explanation.Explanation'>
        # each row corresponds to a sample, each column correspons to a feature
        shap_values = explainer(x_train)
        importances = np.mean(np.abs(shap_values.values),
                              axis=0)  # vertical axis
        ranking = np.argsort(importances)[::-1]

    elif measurement == 'xgboost':
        # XGBClassifier uses gain to measure feature importance as default
        # already normalized
        importances = model.feature_importances_
        # descending order with resepect to feature index
        ranking = np.argsort(importances)[::-1]

    else:
        raise ValueError(
            f"measurement can only take 'shap' or 'xgboost', "
            f"got {measurement} instead.")

    return ranking


def permutation(dataset_name, measurement='xgboost'):
    """Buffer pre-computed feature importance rankings directly in code."""
    if dataset_name == 'cancer':
        if measurement == 'xgboost':
            _permutation = np.array(
                [27, 7, 22, 0, 23, 3, 26, 10, 21, 8, 20, 1, 13, 4, 14, 24, 19, 6,
                 15, 18, 5, 11, 29, 25, 28, 17, 9, 12, 16, 2])
        else:
            pass

    elif dataset_name == 'digits':
        if measurement == 'xgboost':
            _permutation = np.array(
                [42, 54, 53, 20, 6, 60, 43, 30, 51, 12, 27, 33, 46, 10, 22, 5, 4,
                 29, 28, 35, 62, 19, 50, 18, 59, 44, 63, 25, 2, 14, 17, 38, 3, 26,
                 13, 11, 37, 58, 36, 45, 61, 41, 52, 34, 21, 9, 49, 1, 8, 7, 31, 15,
                 16, 23, 24, 32, 39, 40, 47, 48, 55, 56, 57, 0])
        else:
            pass

    elif dataset_name == 'epsilon':
        if measurement == 'xgboost':
            _permutation = np.array(
                [40, 35, 0, 13, 80, 91, 47, 90, 63, 25, 24, 1, 78, 99, 30, 82, 27,
                 53, 89, 22, 48, 36, 21, 9, 44, 2, 28, 59, 75, 67, 74, 70, 88, 68,
                 62, 96, 51, 81, 15, 57, 84, 73, 3, 34, 76, 95, 71, 72, 77, 97, 45,
                 20, 8, 94, 10, 6, 98, 43, 86, 58, 14, 5, 61, 64, 37, 92, 60, 7, 32,
                 23, 11, 38, 83, 56, 39, 33, 19, 87, 42, 31, 49, 29, 50, 17, 65, 18,
                 66, 54, 93, 46, 41, 16, 52, 79, 69, 55, 12, 26, 85, 4])
        else:
            pass

    elif dataset_name == 'census':
        if measurement == 'xgboost':
            _permutation = np.array(
                [12, 13, 2, 32, 22, 14, 34, 1, 33, 40, 3, 19, 31, 35, 26, 25, 9,
                 27, 0, 16, 23, 4, 5, 41, 7, 6, 8, 38, 28, 15, 48, 21, 44, 36,
                 24, 51, 20, 43, 52, 39, 45, 42, 37, 58, 17, 53, 29, 18, 50, 49,
                 59,
                 64, 54, 72, 73, 55, 47, 65, 69, 75, 76, 56, 67, 63, 66, 46, 74,
                 57,
                 61, 78, 77, 30, 10, 11, 71, 70, 68, 60, 62, 79, 80])
        else:
            pass

    elif dataset_name == 'credit':
        if measurement == 'xgboost':
            _permutation = np.array(
                [1, 7, 5, 3, 2, 0, 6, 8, 9, 4])
        else:
            pass

    elif dataset_name == 'default_credit':
        if measurement == 'xgboost':
            _permutation = np.array(
                [18, 19, 20, 21, 22, 0, 13, 9, 1, 14, 8, 10, 15, 2, 7, 6, 12, 3,
                 11, 5, 4, 17, 16])
        else:
            pass

    elif dataset_name == 'diabetes':
        if measurement == 'xgboost':
            _permutation = np.array(
                [8, 2, 3, 7, 9, 1, 6, 5, 4, 0])
        else:
            pass

    elif dataset_name == 'mnist':
        if measurement == 'xgboost':
            _permutation = np.array(
                [67, 70, 101, 740, 220, 583, 743, 709, 358, 100, 96, 104, 720,
                 713, 717, 435, 405, 94, 211, 489, 155, 406, 277, 657, 407, 437,
                 346, 410, 103, 564, 230, 328, 93, 350, 528, 156, 542, 243, 706,
                 569, 235, 514, 290, 487, 124, 329, 745, 163, 710, 570, 567,
                 107,
                 563, 270, 707, 578, 98, 438, 656, 400, 344, 217, 540, 404, 715,
                 343, 473, 276, 556, 432, 597, 126, 190, 543, 512, 360, 294,
                 486,
                 370, 457, 262, 210, 581, 431, 245, 242, 442, 347, 462, 429,
                 178,
                 326, 522, 213, 653, 205, 296, 300, 470, 153, 550, 598, 490,
                 267,
                 596, 453, 511, 517, 327, 544, 353, 164, 315, 318, 428, 177,
                 527,
                 345, 106, 516, 273, 289, 376, 374, 667, 231, 479, 539, 319,
                 379,
                 466, 148, 485, 401, 377, 582, 247, 158, 105, 209, 378, 499,
                 659,
                 151, 179, 367, 600, 95, 426, 638, 317, 455, 624, 464, 389, 269,
                 655, 507, 284, 128, 622, 330, 402, 295, 484, 568, 354, 271,
                 215,
                 298, 409, 375, 342, 538, 551, 708, 302, 381, 268, 458, 515,
                 719,
                 714, 127, 349, 434, 510, 321, 351, 316, 430, 610, 129, 427,
                 320,
                 246, 356, 373, 694, 535, 371, 626, 688, 185, 304, 212, 369,
                 439,
                 403, 408, 433, 606, 660, 712, 322, 333, 536, 460, 383, 203,
                 332,
                 69, 219, 206, 521, 372, 652, 348, 208, 380, 297, 359, 157, 498,
                 463, 154, 65, 654, 323, 690, 523, 132, 266, 546, 123, 658, 705,
                 444, 445, 478, 244, 173, 595, 324, 571, 149, 518, 176, 352,
                 557,
                 165, 579, 482, 620, 681, 399, 218, 683, 650, 133, 311, 272,
                 480,
                 648, 663, 275, 260, 387, 71, 711, 623, 239, 339, 634, 500, 628,
                 134, 331, 651, 301, 108, 150, 186, 293, 591, 425, 605, 488,
                 355,
                 555, 572, 472, 160, 668, 299, 636, 413, 519, 607, 541, 207,
                 357,
                 685, 627, 265, 441, 188, 580, 454, 214, 325, 182, 233, 635,
                 676,
                 398, 291, 382, 684, 661, 386, 664, 468, 633, 680, 414, 524,
                 526,
                 384, 662, 552, 744, 125, 191, 471, 92, 187, 292, 461, 146, 175,
                 147, 411, 174, 388, 456, 573, 689, 741, 746, 508, 119, 491,
                 682,
                 416, 686, 594, 232, 459, 467, 574, 695, 287, 256, 678, 303,
                 513,
                 201, 440, 274, 469, 601, 312, 249, 264, 554, 575, 368, 483,
                 237,
                 238, 474, 415, 181, 183, 603, 314, 263, 200, 637, 240, 687,
                 492,
                 509, 625, 436, 340, 493, 549, 497, 718, 722, 665, 131, 305,
                 553,
                 159, 102, 721, 91, 68, 90, 390, 228, 520, 496, 180, 692, 547,
                 66, 152, 566, 630, 604, 629, 192, 693, 236, 609, 649, 545, 747,
                 288, 677, 189, 608, 202, 184, 742, 122, 341, 417, 481, 222,
                 424,
                 599, 738, 248, 97, 121, 204, 286, 162, 530, 412, 261, 285, 130,
                 494, 166, 385, 443, 135, 592, 216, 397, 666, 576, 74, 548, 632,
                 138, 577, 716, 602, 537, 749, 679, 241, 257, 739, 639, 621,
                 525,
                 465, 691, 495, 234, 613, 136, 396, 258, 501, 99, 145, 221, 640,
                 120, 584, 631, 229, 193, 161, 529, 313, 593, 565, 361, 452,
                 733,
                 612, 611, 171, 44, 737, 283, 137, 619, 117, 259, 734, 73, 250,
                 76, 395, 172, 29, 31, 16, 18, 14, 30, 1, 32, 15, 17,
                 27, 19, 28, 24, 2, 26, 9, 8, 7, 21, 20, 144, 5,
                 143, 10, 11, 12, 13, 4, 23, 22, 25, 3, 6, 55, 33,
                 80, 63, 64, 118, 72, 75, 77, 116, 115, 114, 113, 78, 79,
                 81, 61, 82, 83, 84, 112, 111, 110, 109, 85, 86, 87, 88,
                 89, 62, 60, 142, 43, 141, 140, 139, 34, 35, 36, 37, 38,
                 39, 40, 41, 42, 45, 59, 46, 47, 48, 49, 50, 51, 52,
                 53, 54, 56, 57, 58, 783, 391, 167, 703, 696, 697, 698, 699,
                 700, 701, 702, 704, 589, 723, 724, 725, 726, 727, 728, 729,
                 675,
                 674, 673, 672, 614, 615, 616, 617, 618, 641, 642, 643, 644,
                 645,
                 646, 647, 669, 670, 671, 730, 731, 732, 765, 767, 768, 769,
                 770,
                 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 766,
                 764,
                 735, 763, 736, 748, 750, 751, 752, 753, 754, 755, 756, 757,
                 758,
                 759, 760, 761, 762, 590, 588, 168, 309, 279, 280, 281, 282,
                 306,
                 307, 308, 310, 587, 334, 335, 336, 337, 338, 362, 363, 278,
                 255,
                 254, 253, 169, 170, 194, 195, 196, 197, 198, 199, 223, 224,
                 225,
                 226, 227, 251, 252, 364, 365, 366, 477, 503, 504, 505, 506,
                 531,
                 532, 533, 534, 558, 559, 560, 561, 562, 585, 586, 502, 476,
                 782,
                 475, 392, 393, 394, 418, 419, 420, 421, 422, 423, 446, 447,
                 448,
                 449, 450, 451, 0])
        else:
            pass

    elif dataset_name == 'fashion_mnist':
        if measurement == 'xgboost':
            _permutation = np.array(
                [346, 337, 490, 655, 471, 408, 627, 339, 247, 779, 89, 481, 91,
                 733, 117, 39, 178, 518, 367, 390, 37, 343, 633, 718, 327, 77,
                 206, 499, 742, 549, 145, 592, 281, 315, 352, 267, 207, 403,
                 314,
                 423, 345, 38, 775, 13, 208, 738, 64, 191, 691, 62, 405, 417,
                 173, 92, 175, 119, 595, 745, 425, 232, 389, 406, 495, 219, 205,
                 443, 152, 200, 109, 593, 770, 249, 532, 17, 150, 489, 432, 600,
                 689, 748, 583, 555, 176, 572, 736, 387, 295, 630, 137, 526,
                 695,
                 322, 639, 14, 366, 546, 269, 302, 445, 469, 750, 36, 764, 739,
                 48, 744, 344, 236, 67, 221, 662, 594, 289, 380, 351, 681, 135,
                 714, 331, 41, 270, 317, 580, 557, 115, 434, 210, 301, 204, 283,
                 523, 353, 661, 385, 252, 40, 46, 192, 623, 675, 442, 608, 435,
                 560, 193, 477, 482, 446, 256, 527, 341, 713, 413, 356, 668,
                 160,
                 35, 358, 241, 512, 222, 357, 673, 776, 468, 422, 386, 106, 704,
                 554, 778, 676, 487, 98, 154, 235, 508, 414, 363, 76, 584, 665,
                 622, 185, 679, 227, 153, 162, 165, 182, 347, 667, 87, 613, 752,
                 611, 690, 95, 754, 285, 43, 377, 233, 612, 288, 214, 772, 378,
                 118, 316, 142, 725, 636, 80, 144, 710, 21, 226, 300, 132, 104,
                 723, 573, 190, 120, 298, 228, 407, 303, 194, 418, 134, 253,
                 257,
                 371, 379, 624, 147, 201, 521, 703, 282, 78, 429, 275, 453, 427,
                 767, 123, 362, 727, 126, 309, 651, 777, 696, 70, 381, 308, 460,
                 47, 642, 28, 133, 682, 620, 11, 412, 726, 155, 199, 702, 148,
                 234, 163, 340, 63, 42, 400, 1, 677, 506, 741, 444, 678, 658,
                 720, 707, 409, 181, 276, 517, 774, 640, 372, 342, 10, 540, 348,
                 649, 606, 286, 498, 220, 602, 273, 149, 577, 743, 416, 278,
                 313,
                 421, 44, 246, 335, 124, 231, 626, 596, 274, 384, 65, 599, 359,
                 637, 556, 688, 647, 161, 230, 451, 248, 237, 97, 781, 708, 539,
                 605, 280, 271, 638, 578, 287, 16, 449, 121, 93, 607, 69, 116,
                 45, 537, 609, 188, 529, 245, 20, 272, 719, 125, 565, 765, 7,
                 108, 610, 574, 769, 773, 551, 402, 664, 650, 189, 326, 238,
                 582,
                 552, 648, 462, 369, 321, 399, 255, 652, 15, 330, 260, 766, 217,
                 590, 699, 146, 737, 114, 395, 333, 94, 307, 547, 51, 615, 304,
                 751, 410, 433, 782, 311, 196, 324, 211, 242, 464, 711, 634,
                 753,
                 415, 261, 746, 575, 18, 483, 706, 243, 90, 122, 501, 397, 542,
                 203, 127, 538, 294, 625, 128, 105, 747, 5, 279, 666, 266, 568,
                 455, 334, 732, 504, 50, 657, 216, 567, 763, 541, 143, 509, 440,
                 500, 319, 450, 290, 428, 6, 680, 461, 697, 598, 497, 553, 49,
                 628, 107, 459, 209, 735, 33, 734, 312, 151, 566, 687, 99, 660,
                 174, 663, 621, 262, 244, 653, 717, 9, 79, 72, 374, 515, 284,
                 19, 398, 172, 277, 365, 325, 586, 32, 394, 171, 164, 355, 338,
                 659, 559, 170, 258, 360, 467, 493, 159, 24, 156, 364, 396, 669,
                 12, 519, 229, 438, 463, 441, 757, 494, 259, 759, 693, 780, 264,
                 224, 514, 486, 34, 131, 350, 183, 581, 740, 4, 368, 579, 71,
                 470, 721, 299, 654, 731, 66, 215, 254, 694, 130, 296, 375, 510,
                 591, 218, 31, 762, 56, 430, 530, 585, 528, 564, 177, 291, 113,
                 603, 8, 383, 329, 768, 68, 686, 771, 240, 488, 293, 683, 692,
                 749, 456, 310, 110, 74, 136, 75, 59, 100, 157, 391, 492, 478,
                 268, 525, 543, 102, 709, 496, 715, 635, 263, 52, 761, 411, 96,
                 716, 388, 426, 485, 656, 480, 701, 705, 576, 382, 180, 570,
                 472,
                 601, 373, 239, 328, 724, 439, 454, 618, 479, 722, 401, 265,
                 536,
                 424, 452, 187, 712, 631, 297, 535, 184, 88, 511, 520, 548, 202,
                 61, 318, 129, 465, 491, 755, 571, 457, 507, 54, 3, 619, 370,
                 569, 475, 419, 81, 73, 212, 629, 760, 597, 685, 641, 101, 179,
                 103, 672, 644, 729, 158, 561, 112, 516, 82, 550, 306, 349, 354,
                 544, 522, 447, 484, 404, 305, 524, 292, 545, 531, 466, 84, 186,
                 632, 614, 198, 197, 213, 643, 684, 250, 502, 588, 476, 448, 22,
                 323, 604, 730, 503, 431, 458, 60, 646, 589, 320, 169, 473, 332,
                 513, 392, 23, 437, 436, 167, 376, 558, 166, 563, 223, 698, 83,
                 111, 534, 474, 361, 562, 336, 30, 616, 53, 758, 505, 420, 139,
                 26, 533, 29, 670, 58, 393, 700, 728, 251, 138, 587, 645, 141,
                 617, 674, 195, 140, 85, 671, 86, 225, 55, 57, 2, 783, 25,
                 27, 168, 756, 0])
        else:
            pass

    else:
        raise ValueError(f"np_dataset '{dataset_name}'' not supported.")

    return _permutation