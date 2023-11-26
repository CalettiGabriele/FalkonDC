import sys
import numpy as np
import torch

N = str(sys.argv[1])
mode = str(sys.argv[2])
out = []


def rmse(true, pred):
    return torch.sqrt(torch.mean((true.reshape(-1, 1) - pred.reshape(-1, 1)) ** 2))


def multi_acc(true, pred):
    out = []
    for i in range(len(pred)):
        out.append(true[i] == pred[i])
        # print(true[i], "    <--->   ", pred[i])
    return int(sum(out)) * 100 / len(out)


def acc(true, pred):
    out = []
    for i in range(len(pred)):
        if pred[i] > 0:
            out.append(true[i] == 1)
        else:
            out.append(true[i] == -1)
    return int(sum(out)) * 100 / len(out)

def acc(true, pred):
    out = []
    for i in range(len(pred)):
        if pred[i] >= 0.5:
            out.append(true[i] == 1)
        elif pred[i] <= -0.5:
            out.append(true[i] == -1)
        else:
            out.append(true[i] == 0)
    return int(sum(out)) * 100 / len(out)


def aggregate(*arg):

    temp = []
    for i in range(len(arg[0])):
        count = 0
        for el in arg:
            count += el[i]
        temp.append(count)

    temp = np.array(temp)
    if temp.argmax() == 0:
        return [1, 0, 0]
    elif temp.argmax() == 1:
        return np.array([0, 1, 0])
    return [0, 0, 1]


Y_test = torch.load("./Y_test.pt")
for p in range(int(N)):
    name = "Pred_" + str(p)
    locals()[name] = torch.load("./Predictions/Pred_" + str(p) + ".pt")
for i in range(len(Y_test)):
    summ = 0
    files = []
    for j in range(int(N)):
        name = "Pred_" + str(j)
        summ += locals()[name][i]
        files.append(locals()[name][i])
    if mode == "-c":  # for binary classification problems
        out.append(summ)
    elif mode == "-r":
        out.append(summ / int(N))
    elif mode == "-m":
        out.append(summ)
    #out.append(aggregate(*files))

if mode == "-c":
    print("--> ACCURACY: ", acc(Y_test, out), "%")
    # print("Test BinaryLoss: %.3f" % (binary_loss(Y_test, out)))
elif mode == "-r":
    print("Test RMSE: %.3f" % (rmse(Y_test, torch.FloatTensor(out))))
elif mode == "-m":
    print("--> ACCURACY: ", accM(Y_test, out), "%")
    '''print(
        "--> ACCURACY: ",
        multi_acc(Y_test, torch.from_numpy(np.array(out)).to(dtype=torch.float32)),
        "%",
    )'''
else:
    print("Not Valid Parameter!")
