import sys
import numpy as np
import torch
import falkon

N = str(sys.argv[1])


def rmse(true, pred):
    return torch.sqrt(torch.mean((true.reshape(-1, 1) - pred.reshape(-1, 1)) ** 2))


X_train = torch.load("/root/Data/X_train_" + N + ".pt")
X_test = torch.load("/root/Data/X_test.pt")
Y_train = torch.load("/root/Data/Y_train_" + N + ".pt")

options = falkon.FalkonOptions(keops_active="no", use_cpu=True)
kernel = falkon.kernels.GaussianKernel(sigma=1, opt=options)
flk = falkon.Falkon(kernel=kernel, penalty=1e-5, M=1000, options=options)

flk.fit(X_train, Y_train)

pred = flk.predict(X_test)
# print(pred)

torch.save(torch.sign(pred).to(torch.float32), "/root/Pred_" + N + ".pt")
