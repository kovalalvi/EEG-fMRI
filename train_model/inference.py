from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np
import torch

def evaluation(model, test_loader):
    """Suppose that batch size  = len(test_data)
    return: vector of predictions and true value in numpy arrays
    """
    device = 'cpu'
    model.to(device)
    model.eval()
    batch_size = test_loader.batch_size
    size = len(test_loader) * batch_size
    yhat = torch.zeros([0, 1, 1 ])
    y = torch.zeros([0, 1 , 1])
    with torch.no_grad():
        counter = 0
        for x_batch, y_batch in test_loader:
            counter += 1
            out = model(x_batch)
            yhat = torch.cat((yhat, out[:, -1:]), dim = 0)
            y = torch.cat((y, y_batch[:, -1:]), dim = 0)
        yhat = yhat.reshape(-1).numpy()
        y = y.reshape(-1).numpy()
    return yhat, y


def mse_loss(y, yhat):
    return np.mean((y-yhat)**2)
def show_metrics(y, y_hat):
    print('Size of test data', y.shape, y_hat.shape)
    return r2_score(y, y_hat), pearsonr(y, y_hat)[0], mse_loss(y, y_hat)
