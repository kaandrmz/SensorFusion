import numpy as np



def confusion_matrix_aggre(x, y, n, ignore_label=None, mask=None):
    """Compute confusion matrix

    Args:
        x (np.array): 1 x h x w
            prediction array
        y (np.array): 1 x h x w
            groundtruth array
        n (int): number of classes
        ignore_label (int, optional): index of ignored label. Defaults to None.
        mask (np.array, optional): mask of regions that is needed to compute. Defaults to None.

    Returns:
        np.array: n x n
            confusion matrix
    """
    if mask is None:
        mask = np.ones_like(x) == 1
    x[x==ignore_label] = y[x==ignore_label]
    # global unlabel_num
    unlabel_num = sum(x==ignore_label)
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    # k = (x >= 0) & (y < n) & (mask.astype(np.bool))
    #print(k)
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n ** 2).reshape(n, n),unlabel_num


def getFreq_aggre(conf_matrix,unlabel_sum):
    """Compute frequentice of each class

    Args:
        conf_matrix (np.array): n x n
            confusion matrix

    Returns:
        np.array: (n, )
            frequentices of classes
    """
    if conf_matrix.sum() == 0:
        return 0
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    return freq


def getIoU_aggre(conf_matrix,unlabel_sum):
    """Compute IoU

    Args:
        conf_matrix (np.array): n x n
            confusion matrix

    Returns:
        np.array: (n,)
            IoU of classes
    """
    if conf_matrix.sum() == 0:
        return 0
    with np.errstate(divide="ignore", invalid="ignore"):
        union = np.maximum(1.0, conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))-unlabel_sum
        intersect = np.diag(conf_matrix)-unlabel_sum
        IU = np.nan_to_num(intersect / union)
    return IU

def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    """Compute confusion matrix

    Args:
        x (np.array): 1 x h x w
            prediction array
        y (np.array): 1 x h x w
            groundtruth array
        n (int): number of classes
        ignore_label (int, optional): index of ignored label. Defaults to None.
        mask (np.array, optional): mask of regions that is needed to compute. Defaults to None.

    Returns:
        np.array: n x n
            confusion matrix
    """
    if mask is None:
        mask = np.ones_like(x) == 1
    # x[x==ignore_label] = y[x==ignore_label]
    # global unlabel_num
    # unlabel_num = sum(x==ignore_label)
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    # k = (x >= 0) & (y < n) & (mask.astype(np.bool))
    #print(k)
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n ** 2).reshape(n, n)


def getIoU(conf_matrix):
    """Compute IoU

    Args:
        conf_matrix (np.array): n x n
            confusion matrix

    Returns:
        np.array: (n,)
            IoU of classes
    """
    if conf_matrix.sum() == 0:
        return 0
    with np.errstate(divide="ignore", invalid="ignore"):
        union = np.maximum(1.0, conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))
        # union = np.maximum(1.0,conf_matrix.sum(axis=0))
        intersect = np.diag(conf_matrix)
        IU = np.nan_to_num(intersect / union)
        # print(IU)
    return IU


def getFreq(conf_matrix):
    """Compute frequentice of each class

    Args:
        conf_matrix (np.array): n x n
            confusion matrix

    Returns:
        np.array: (n, )
            frequentices of classes
    """
    if conf_matrix.sum() == 0:
        return 0
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    return freq


def get_mean_iou(conf_mat, dataset,label_num=None):
    """Get mean IoU for each different dataset

    Args:
        conf_mat (np.array): n x n
            confusion matrix
        dataset (str): dataset name

    Returns:
        float: mean IoU
    """
    IoU = getIoU(conf_mat)
    if dataset == "deepglobe":
        return np.nanmean(IoU[1:])
    elif dataset == "cityscapes":
        return np.nanmean(IoU)
    elif dataset == "ade20k" or dataset == "cocostuff" or dataset == "pascal":
        if label_num is not None:
            return np.nansum(IoU)/label_num
        else:
            return np.nanmean(IoU)
    else:
        raise "Not implementation for dataset {}".format(dataset)


def get_freq_iou(conf_mat, dataset):
    """Get frequent IoU for each different dataset

    Args:
        conf_mat (np.array): n x n
            confusion matrix
        dataset (str): dataset name

    Returns:
        float: frequent IoU
    """
    IoU = getIoU(conf_mat)
    freq = getFreq(conf_mat)
    if dataset == "deepglobe":
        return (IoU[1:] * freq[1:]).sum() / freq[1:].sum()
    else:
        if type(IoU) is int:
            return 0
        else:
            return (IoU * freq).sum()


def get_overall_iou(conf_mat, dataset,label_num=None):
    """Get overall IoU for each different dataset

    Args:
        conf_mat (np.array): n x n
            confusion matrix
        dataset (str): dataset name

    Returns:
        float: overall iou
    """
    if dataset in ["deepglobe", "cityscapes","ade20k","cocostuff","pascal"]:
        return get_mean_iou(conf_mat, dataset,label_num)
    else:
        raise "Not implementation for dataset {}".format(dataset)
