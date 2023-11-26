import sys
from sklearn import datasets, model_selection
import torch
import falkon
import tracemalloc
import GlobalPreconditioner

N = str(sys.argv[1])

if N==2 or N==5 or N==8:
    m=5000
elif N==1 or N==4 or N==7:
    m=1000
else:
    m=500


tracemalloc.start()

X_train = torch.load("/root/Data/X_train_" + N + ".pt")
X_test = torch.load("/root/Data/X_test.pt")
Y_train = torch.load("/root/Data/Y_train_" + N + ".pt")

flk_opt = falkon.FalkonOptions(use_cpu=True, keops_active="no")

flk_kernel = falkon.kernels.GaussianKernel(1, opt=flk_opt)

flk = falkon.Falkon(kernel=flk_kernel, penalty=1e-7, M=m, options=flk_opt)

flk.fit(X_train, Y_train)

flk_pred = flk.predict(X_test)

torch.save(torch.sign(flk_pred).to(torch.float32), "/root/Pred_" + N + ".pt")

print("### Memory: ", tracemalloc.get_traced_memory())
tracemalloc.stop()
