import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def Train(model, trdl, cri, op, EPOCH):
    lh = []
    n = len(trdl.dataset)
    model.train()
    for ep in range(EPOCH):
        rloss = 0
        for x, y in trdl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            yhat = model(x)
            loss = cri(yhat, y)
            op.zero_grad()
            loss.backward()
            op.step()
            lb = loss.item() * x.shape[0]
            rloss += lb
        losse = rloss / n
        lh += [losse]
        print(f"epoch: {ep+1}, train loss: {losse:.3f}")
    return lh


def Test(model, tedl):
    model.eval()
    with torch.no_grad():
        rcorrect = 0
        for x, y in tedl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            yhat = model(x)
            pred = yhat.argmax(dim=1)
            correctsb = torch.sum(pred == y).item()
            rcorrect += correctsb
        accuracye = rcorrect / len(tedl.dataset) * 100
    print(f"test accuracy: {rcorrect}/{len(tedl.dataset)} {accuracye:.1f}%")
    return round(accuracye, 1)


def Acc(m, d, n):
    m.eval()
    with torch.no_grad():
        rc = 0
        for x, y in d:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            yh = m(x)
            pr = yh.argmax(dim=1)
            cr = torch.sum(pr == y).item()
            rc += cr
        acce = rc / len(d.dataset) * 100
    print(f"{n} accuracy: {rc}/{len(d.dataset)} ({acce:.1f}%)")
    return round(acce, 1)


def Test_plot(model, tedl):
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(tedl))
        xb = xb.to(DEVICE)
        yhat = model(xb)
        pred = yhat.argmax(dim=1)
    xb = xb.to("cpu")
    plt.figure(figsize=(8, 4))
    for i in range(6):
        plt.subplot(2, 3, i + 1, xticks=[], yticks=[])
        plt.imshow(xb[i].permute(1, 2, 0), cmap="gray")
        pred_class = tedl.dataset.classes[pred[i]]
        true_class = tedl.dataset.classes[yb[i]]
        plt.title(
            f"{pred_class} ({true_class})",
            color="g" if pred_class == true_class else "r",
        )


def count_params(model):
    num = sum([i.numel() for i in model.parameters() if i.requires_grad])
    return num


def get_conf(model, tedl):
    n = len(tedl.dataset.classes)
    model.eval()
    with torch.no_grad():
        confusion = torch.zeros(n, n)
        for xb, yb in tedl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            yhat = model(xb)
            pred = yhat.argmax(dim=1)
            confusion += torch.bincount(n * yb + pred.cpu(), minlength=n**2).reshape(
                n, n
            )
        confusion = confusion.numpy()
    return confusion


def plot_conf_mat(confusion, classes=None):
    n = confusion.shape[0]
    acc = np.trace(confusion) / np.sum(confusion) * 100
    plt.figure(figsize=(10, 7))
    plt.imshow(confusion, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for i in range(n):
        for j in range(n):
            plt.text(
                j,
                i,
                confusion[i, j],
                ha="center",
                va="center",
                color="white" if confusion[i, j] > confusion.max() / 1.5 else "black",
            )
        if classes is not None:
            plt.xticks(range(n), classes)
            plt.yticks(range(n), classes)
        else:
            plt.xticks(range(n))
            plt.yticks(range(n))
        plt.xlabel(f"perdicted label\n accuracy={acc:.1f}%")
        plt.ylabel("true label")
