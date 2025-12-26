import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


def acp(x: np.ndarray, ddof=0, scal=True):
    n, m = x.shape
    x_ = x - np.mean(x, axis=0)
    if scal:
        x_ = x_ / np.std(x, axis=0, ddof=ddof)
    r_v = (1 / (n - ddof)) * x_.T @ x_
    valp,vecp = np.linalg.eig(r_v)
    # print(vecp)
    # print(valp)
    k = np.flip(np.argsort(valp))
    # print(k)
    alpha = valp[k]
    # print(alpha)
    a = vecp[:,k]
    return x_,r_v,alpha,a

def tabelare_varianta(alpha:np.ndarray):
    m = len(alpha)
    t = pd.DataFrame(index=["C"+str(i+1) for i in range(m)])
    t["Varianta"] = alpha
    t["Varianta cumulata"] = np.cumsum(alpha)
    varianta_totala = sum(alpha)
    t["Procent varianta"] = alpha*100/varianta_totala
    t["Procent cumulat"] = np.cumsum(t["Procent varianta"])
    return t


def salvare_ndarray(x:np.ndarray,nume_linii,nume_coloane,nume_index="",nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.index.name=nume_index
    temp.to_csv(nume_fisier_output)
    return temp

