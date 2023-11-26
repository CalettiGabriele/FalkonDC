import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import torch

file_name = str(sys.argv[1])
print("##############################", file_name)
mode = str(sys.argv[2])
samples = int(sys.argv[3])  # total number of elements in the dataset
P = int(sys.argv[4])  # number of processes and sub-sets
O = int(sys.argv[5])  # O between sub-sets
cl = '-m'
N_train = int(samples * 0.9)  # number of elements for training-set
N_test = int(samples * 0.1)  # number of elements for test-set

# import the dataset and shuffle it
df = pd.read_csv(file_name)
df = df.sample(frac=1)
print("-->  ", file_name, "\n-->  ", len(df), "\n-->    ", df.loc[0])

# split X and Y data
Y_data = df.iloc[:, -1:].to_numpy()  # .reshape(-1, 1)
X_data = df.iloc[:, :-1].to_numpy()
print("<<<<<<<<< ", X_data)
# split train and test sets
X_train = torch.from_numpy(X_data[:N_train]).to(dtype=torch.float32)
X_test = torch.from_numpy(X_data[N_train : N_train + N_test]).to(dtype=torch.float32)
Y_train = torch.from_numpy(Y_data[:N_train]).to(dtype=torch.float32)
Y_test = torch.from_numpy(Y_data[N_train : N_train + N_test]).to(dtype=torch.float32)

train_mean = X_train.mean(0, keepdim=True)
train_std = X_train.std(0, keepdim=True)
X_train -= train_mean
X_train /= train_std
X_test -= train_mean
X_test /= train_std


if mode == "--split" or mode == "-s":
    # split and save P sub-sets
    dim = int(len(X_train) / P)
    end = dim
    for i in range(P):
        if i == 0:
            torch.save(X_train[: end + O], "X_train_" + str(i) + ".pt")
            torch.save(Y_train[: end + O], "Y_train_" + str(i) + ".pt")
            print(
                "<<<<<<<<<<<<<<< X_train_" + str(i), " Dim: ", len(X_train[: end + O])
            )
        elif i == P - 1:
            torch.save(X_train[start - O :], "X_train_" + str(i) + ".pt")
            torch.save(Y_train[start - O :], "Y_train_" + str(i) + ".pt")
            print(
                "<<<<<<<<<<<<<<< X_train_" + str(i), " Dim: ", len(X_train[start - O :])
            )
        else:
            torch.save(X_train[start - O : end + O], "X_train_" + str(i) + ".pt")
            torch.save(Y_train[start - O : end + O], "Y_train_" + str(i) + ".pt")
            print(
                "<<<<<<<<<<<<<<< X_train_" + str(i),
                " Dim: ",
                len(X_train[start - O : end + O]),
            )
        start = end
        end += dim
elif mode == "--partition" or mode == "-p":
    kmeans = KMeans(n_clusters=P, random_state=0, n_init="auto").fit(X_train)
    for i in range(P):
        X_subset = []
        Y_subset = []
        for idx, el in enumerate(kmeans.labels_):
            if el == i:
                X_subset.append(X_train[idx])
                Y_subset.append(Y_train[idx])
        torch.save(
            torch.from_numpy(np.array(X_subset)).to(dtype=torch.float32),
            "X_train_" + str(i) + ".pt",
        )
        torch.save(
            torch.from_numpy(np.array(Y_subset)).to(dtype=torch.float32),
            "Y_train_" + str(i) + ".pt",
        )
        print("<<<<<<<<<<<<<<< X_train_" + str(i), " Dim: ", len(X_subset))
elif mode == "-c":
    dim = int(len(X_train)/18)
    torch.save(X_train[: dim*3], "X_train_0.pt")
    torch.save(Y_train[: dim*3], "Y_train_0.pt")
    torch.save(X_train[dim*3: dim*6], "X_train_1.pt")
    torch.save(Y_train[dim*3: dim*6], "Y_train_1.pt")
    torch.save(X_train[dim*6: dim*9], "X_train_2.pt")
    torch.save(Y_train[dim*6: dim*9], "Y_train_2.pt")
    torch.save(X_train[dim*9: dim*11], "X_train_3.pt")
    torch.save(Y_train[dim*9: dim*11], "Y_train_3.pt")
    torch.save(X_train[dim*11: dim*13], "X_train_4.pt")
    torch.save(Y_train[dim*11: dim*13], "Y_train_4.pt")
    torch.save(X_train[dim*13: dim*15], "X_train_5.pt")
    torch.save(Y_train[dim*13: dim*15], "Y_train_5.pt")
    torch.save(X_train[dim*15: dim*16], "X_train_6.pt")
    torch.save(Y_train[dim*15: dim*16], "Y_train_6.pt")
    torch.save(X_train[dim*16: dim*17], "X_train_7.pt")
    torch.save(Y_train[dim*16: dim*17], "Y_train_7.pt")
    torch.save(X_train[dim*17: ], "X_train_8.pt")
    torch.save(Y_train[dim*17: ], "Y_train_8.pt")

torch.save(X_test, "X_train.pt")
torch.save(Y_test, "Y_train.pt")
torch.save(X_test, "X_test.pt")
torch.save(Y_test, "Y_test.pt")
