import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
import plotly.express as px
import os

def plot_varianta(alpha:np.ndarray,procent_minimal=80,scal=True):
    m = len(alpha)
    x = np.arange(1,m+1)
    f = plt.figure(figsize=(8,5))
    ax = f.add_subplot(1,1,1)
    ax.set_title("Plot varianta componente",fontdict={"color":"b","fontsize":16})
    ax.plot(x,alpha)
    ax.set_xlabel("Componente",fontsize=12)
    ax.set_ylabel("Varianta",fontsize=12)
    ax.set_xticks(x)
    ax.scatter(x,alpha,c="r")
    k1 = None
    if scal:
        ax.axhline(1,c="g",label="Criteriul Kaiser")
        valori_1 = np.where(alpha>1)
        k1 = len(valori_1[0])
    procent_cumulat = np.cumsum(alpha*100/np.sum(alpha))
    k2 = np.where( procent_cumulat > procent_minimal)[0][0]+1
    ax.axhline(alpha[k2-1],c="m",label="Acoperire minimala ("+str(procent_minimal)+")")
    eps = alpha[:m-1] - alpha[1:]
    sigma = eps[:m-2] - eps[1:]
    # print(sigma)
    k3=None
    exista_negative = any(sigma<0)
    if exista_negative:
        k3 = np.where(sigma<0)[0][0]+2
        ax.axhline(alpha[k3-1], c="c", label="Criteriul Cattell")
    ax.legend()
    plt.savefig("graphics/graphics_acp/plot_varianta_componente.png")
    return k1,k2,k3

def show():
    plt.show()

def corelograma(t:pd.DataFrame,titlu="corelograma",vmin=-1,vmax=1,cmap = "RdYlBu",annot=True):
    f = plt.figure(figsize=(12, 10))  # Mărește puțin figura
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"color": "b", "fontsize": 16})
    heatmap(t, vmin=vmin, vmax=vmax, cmap=cmap, annot=annot, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("graphics/graphics_acp/"+titlu+".png")


def plot_scoruri_corelatii(t: pd.DataFrame,
                           varx="C1",
                           vary="C2",
                           titlu="Plot scoruri",
                           etichete=None,
                           corelatii=False,
                           top_outlieri=20  # <- control clar
                           ):
    f = plt.figure(figsize=(14, 10))
    # aspect=1 doar pentru cercul corelațiilor
    ax = f.add_subplot(1, 1, 1, aspect=1 if corelatii else "auto")

    ax.set_title(titlu, fontdict={"color": "b", "fontsize": 16})
    ax.set_xlabel(varx)
    ax.set_ylabel(vary)

    if corelatii:
        pas = 0.05
        theta = np.arange(0, np.pi * 2 + pas, pas)
        ax.plot(np.cos(theta), np.sin(theta))
        ax.plot(0.7 * np.cos(theta), 0.7 * np.sin(theta), c="g")

    # toate punctele
    ax.scatter(t[varx], t[vary], c="r", alpha=0.5, s=30)

    # axele la 0 (ca în seminar)
    ax.axvline(0, c="k")
    ax.axhline(0, c="k")

    if etichete is not None:
        # distanța față de origine (outlieri)
        distante = t[varx] ** 2 + t[vary] ** 2
        idx_out = distante.sort_values(ascending=False).head(top_outlieri).index

        romania_plotted = False

        for i in range(len(t)):
            idx = t.index[i]
            nume = str(etichete[i]).strip()

            if (idx in idx_out) or (nume == "Romania"):
                x_val = t[varx].iloc[i]
                y_val = t[vary].iloc[i]
                ax.text(x_val, y_val, nume, fontsize=9, color="black")

                if nume == "Romania":
                    ax.scatter(x_val, y_val, c="blue", s=100, zorder=10, label="Romania")
                    romania_plotted = True

        if romania_plotted:
            ax.legend()

    plt.tight_layout()
    plt.savefig("graphics/graphics_acp/" + titlu + "-" + varx + "-" + vary + ".png")


def harta_scoruri(t_scoruri: pd.DataFrame, t_initial: pd.DataFrame, coloana_iso="ISO_Code", titlu="Harta Scoruri"):
    if not os.path.exists("graphics/graphics_acp/"):
        os.makedirs("graphics/graphics_acp/")

    if coloana_iso not in t_initial.columns:
        print(f"EROARE: Nu gasesc coloana '{coloana_iso}' in datele initiale!")
        return

    df_harta = t_initial[[coloana_iso]].join(t_scoruri, how="inner")
    df_harta = df_harta.reset_index()

    nume_vechi = df_harta.columns[0]
    df_harta.rename(columns={nume_vechi: "Countries"}, inplace=True)

    coloane_componente = [c for c in t_scoruri.columns if c.startswith("C")]

    for comp in coloane_componente:
        fig = px.choropleth(
            df_harta,
            locations=coloana_iso,
            color=comp,
            hover_name="Countries",  # Acum stim sigur ca exista coloana asta
            color_continuous_scale="RdYlBu",
            title=f"{titlu} - {comp}",
            template="plotly_white"
        )

        fig.update_layout(
            coloraxis_colorbar=dict(title="Scor"),
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )

        out_path = f"graphics/graphics_acp/{titlu}_{comp}.html"
        fig.write_html(out_path)
        print(f"-> Harta salvata: {out_path}")