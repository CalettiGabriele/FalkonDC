import sys
from sklearn import datasets, model_selection
import torch
import falkon

N = str(sys.argv[1])


def binary_loss(true, pred):
    return torch.mean((true != torch.sign(pred)).to(torch.float32))


X_train = torch.load("/root/Data/X_train_" + N + ".pt")
X_test = torch.load("/root/Data/X_test.pt")
Y_train = torch.load("/root/Data/Y_train_" + N + ".pt")

logflk_opt = falkon.FalkonOptions(use_cpu=True)

logflk_kernel = falkon.kernels.GaussianKernel(1, opt=logflk_opt)
logloss = falkon.gsc_losses.LogisticLoss(logflk_kernel)

penalty_list = [1e-3, 1e-5, 1e-7, 1e-7, 1e-7]
iter_list = [4, 4, 4, 8, 8]

logflk = falkon.LogisticFalkon(
    kernel=logflk_kernel,
    penalty_list=penalty_list,
    iter_list=iter_list,
    M=10000,
    loss=logloss,
    error_fn=binary_loss,
    error_every=1,
    options=logflk_opt,
)

logflk.fit(X_train, Y_train)

logflk_pred = logflk.predict(X_test)

torch.save(torch.sign(logflk_pred).to(torch.float32), "/root/Pred_" + N + ".pt")
