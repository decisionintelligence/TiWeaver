import numpy as np
import torch


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true + 1e-10)))


def SMAPE(pred, true):
    # Avoid division by zero by adding a small constant
    denominator = (torch.abs(true) + torch.abs(pred)) / 2 + 1e-8

    # Calculate the SMAPE
    smape_value = torch.mean(torch.abs(pred - true) / denominator)

    return smape_value

def MSE_with_mask(x, x_hat, mask):
    return np.power((x - x_hat) * mask, 2).sum() / (mask.sum())

def MAE_with_mask(x, x_hat, mask):
    return np.abs((x - x_hat) * mask).sum() / (mask.sum())

def MAPE_with_mask(x, x_hat, mask):
    return np.abs(100 * [(x - x_hat) / (x_hat + 1e-10)] * mask).sum() / (mask.size)



def masked_loss(y_pred, y_true, loss_func):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()  # assign the sample weights of zeros to nonzero-values
    loss = loss_func(y_pred, y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_rmse_loss(y_pred, y_true):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_pred - y_true, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())


def compute_all_metrics(y_pred, y_true):
    # mae = masked_loss(y_pred, y_true, MAE).item()
    # rmse = masked_rmse_loss(y_pred, y_true).item()
    # smape = masked_loss(y_pred, y_true, SMAPE).item()
    mae = MAE(y_pred, y_true)
    mse = MSE(y_pred, y_true)
    mape = MAPE(y_pred, y_true)
    return mae, mse, mape

def compute_all_metrics_with_mask(y_pred, y_true, y_mask):
    # mae = masked_loss(y_pred, y_true, MAE).item()
    # rmse = masked_rmse_loss(y_pred, y_true).item()
    # smape = masked_loss(y_pred, y_true, SMAPE).item()
    mae = MAE_with_mask(y_pred, y_true, y_mask)
    mse = MSE_with_mask(y_pred, y_true, y_mask)
    mape = MAPE_with_mask(y_pred, y_true, y_mask)
    return mae, mse, mape


# def MAE(pred, true):
#     return np.mean(np.abs(pred - true))

if __name__ == '__main__':
    y_pred = np.array([[[1, 2, 3], [4, 5, 6]],
                       [[1, 2, 3], [4, 5, 6]]])
    y_true = np.array([[[1.1, 2.2, 3.3], [3.9, 5.4, 6.5]],
                       [[1.2, 2.2, 3.1], [4, 5.2, 6.15]]])

    mae = MAE(y_pred, y_true)
    print(mae)

    # mae, smape, rmse = compute_all_metrics(y_pred, y_true)
    # print(mae, masked_loss(y_pred, y_true, MAE))
    # print(rmse, masked_rmse_loss(y_pred, y_true))
    # print(smape, masked_loss(y_pred, y_true, SMAPE))
