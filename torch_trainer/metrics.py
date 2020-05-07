import torch


def accuracy(y_pred, y_true):
    num_corrects = (torch.max(y_pred, 1)[1].view(
        y_true.size()).data == y_true.data).float().sum()
    score = 100.0 * num_corrects / len(y_true)
    return score
