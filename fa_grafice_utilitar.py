import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from seaborn import heatmap
import os
import plotly.express as px

def show():
    plt.show()

def corelograma(t:pd.DataFrame,titlu="Corelograma",vmin=-1,vmax=1,cmap = "RdYlBu",annot=True):
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"color": "b", "fontsize": 16})
    heatmap(t,vmin=vmin,vmax=vmax,cmap=cmap,annot=annot,ax=ax)

    # +++ MINIM (ca să nu se taie textul) +++
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("graphics/graphics_fa/"+titlu+".png", bbox_inches="tight", pad_inches=0.2, dpi=300)

def plot_scoruri_corelatii(t:pd.DataFrame,
                           varx="C1",
                           vary = "C2",
                           titlu="Plot scoruri",
                           etichete=None,
                           corelatii=False
                        ):
    f = plt.figure(figsize=(10, 6))
    ax = f.add_subplot(1, 1, 1, aspect=1)
    ax.set_title(titlu, fontdict={"color": "b", "fontsize": 16})
    ax.set_xlabel(varx)
    ax.set_ylabel(vary)
    if corelatii:
        pas = 0.05
        theta = np.arange(0,np.pi*2+pas,pas)
        ax.plot(np.cos(theta),np.sin(theta))
        ax.plot(0.7*np.cos(theta),0.7*np.sin(theta),c="g")
    ax.scatter(t[varx],t[vary],c="r",alpha=0.5)
    ax.axvline(0,c="k")
    ax.axhline(0,c="k")
    if etichete is not None:
        n = len(t)
        for i in range(n):
            ax.text(t[varx].iloc[i],t[vary].iloc[i],etichete[i])

    # +++ MINIM (ca să nu se taie titlul/etichetele) +++
    plt.tight_layout()
    plt.savefig("graphics/graphics_fa/"+titlu+"-"+varx+"-"+vary+".png", bbox_inches="tight", pad_inches=0.2, dpi=300)


def harta_scoruri_factoriale(t_scoruri: pd.DataFrame,
                             t_initial: pd.DataFrame,
                             coloana_iso="ISO_Code",
                             titlu="Harta scoruri factoriale"):
    os.makedirs("graphics/graphics_fa", exist_ok=True)

    if coloana_iso not in t_initial.columns:
        print(f"EROARE: Nu gasesc coloana '{coloana_iso}' in datele initiale!")
        return

    # t_initial: index = Countries (ideal), contine ISO_Code
    df_harta = t_initial[[coloana_iso]].join(t_scoruri, how="inner").reset_index()

    # siguranta: primul nume de coloana e tara (indexul)
    nume_vechi = df_harta.columns[0]
    df_harta.rename(columns={nume_vechi: "Countries"}, inplace=True)

    coloane_factori = [c for c in t_scoruri.columns if c.startswith("F")]

    for fct in coloane_factori:
        fig = px.choropleth(
            df_harta,
            locations=coloana_iso,
            color=fct,
            hover_name="Countries",
            color_continuous_scale="RdYlBu",
            title=f"{titlu} - {fct}",
            template="plotly_white"
        )

        fig.update_layout(
            coloraxis_colorbar=dict(title="Scor"),
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )

        out_path = f"graphics/graphics_fa/{titlu}_{fct}.html"
        fig.write_html(out_path)
        # fig.write_image(f"graphics/graphics_fa/{titlu}_{fct}.png", width=1200, height=700)
        print(f"-> Harta salvata: {out_path}")
