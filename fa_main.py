import os
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import factor_analyzer as fa
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


pd.set_option("display.max_columns", None)
np.set_printoptions(3, sys.maxsize, suppress=True)


# ------------------------------------------------------------------------------
# UTILITAR: inlocuire lipsuri
# ------------------------------------------------------------------------------
def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


def main():
    # --------------------------------------------------------------------------
    # foldere output
    # --------------------------------------------------------------------------
    os.makedirs("data_out/data_out_fa", exist_ok=True)

    # --------------------------------------------------------------------------
    # citire date
    # --------------------------------------------------------------------------
    t = pd.read_csv("data_in/economicdata2023-2023.csv")
    nan_replace_df(t)

    variabile_observate = list(t)[2:]

    # setam Countries ca index (pentru etichete)
    t.set_index("Countries", inplace=True)

    # >>> AICI se defineste x (matricea numerica) <<<
    x = t[variabile_observate].values

    # --------------------------------------------------------------------------
    # A. Test Bartlett + validare model
    # --------------------------------------------------------------------------
    test_bartlett = fa.calculate_bartlett_sphericity(x)
    chi2_stat, p_value = float(test_bartlett[0]), float(test_bartlett[1])

    t_bartlett = pd.DataFrame(
        data={"Chi2": [chi2_stat], "p_value": [p_value]},
        index=["Bartlett"]
    )
    t_bartlett.index.name = "Test"
    t_bartlett.to_csv("data_out/data_out_fa/bartlett.csv")

    print("Test Bartlett (sfericitate)")
    print("Chi2 =", chi2_stat)
    print("p-value =", p_value)

    if p_value > 0.001:
        print("Nu exista factori comuni! Modelul factorial este RESPINS (p-value > 0.001).")
        sys.exit(0)

    print("Exista factori comuni! Modelul factorial este VALID (p-value <= 0.001).")

    # --------------------------------------------------------------------------
    # B1. Indecsi KMO
    # --------------------------------------------------------------------------
    kmo = fa.calculate_kmo(x)

    t_kmo = pd.DataFrame(
        data={"KMO": np.append(kmo[0], kmo[1])},
        index=variabile_observate + ["Total"]
    )
    t_kmo.index.name = "Indicator"
    t_kmo.to_csv("data_out/data_out_fa/kmo.csv")

    print("\nIndecsi KMO:")
    print(t_kmo)

    # ------------------------------------------------------------------------------
    # B2. ANALIZA VARIANTEI FACTORILOR
    # ------------------------------------------------------------------------------

    model_af = fa.FactorAnalyzer(n_factors=x.shape[1], rotation="varimax")
    model_af.fit(x)
    varianta = model_af.get_factor_variance()

    # Construim tabelul de varianta
    t_varianta = pd.DataFrame(
        data={
            "Varianta": varianta[0],
            "Varianta cumulata": np.cumsum(varianta[0]),
            "Procent varianta": varianta[1] * 100,
            "Procent cumulat": varianta[2] * 100
        },
        index=["F" + str(i + 1) for i in range(len(varianta[0]))]
    )

    # Salvare tabel
    t_varianta.index.name = "Factor"
    t_varianta.to_csv("data_out/data_out_fa/Varianta.csv")

    # Afisare pentru verificare
    print("\nVarianta factorilor comuni:")
    print(t_varianta.round(3))

    # ------------------------------------------------------------------------------
    # B3. CORELATII VARIABILE OBSERVATE - FACTORI (LOADINGS)
    #     CU si FARA ROTATIE
    # ------------------------------------------------------------------------------

    # Alegem numarul de factori pentru afisare/analiza
    # Conform output-ului tau, Kaiser da 2 factori (F1, F2).
    nr_factori = 2

    etichete_factori = ["F" + str(i) for i in range(1, nr_factori + 1)]

    # --- 1) Model FARA rotatie ---
    model_af_0 = fa.FactorAnalyzer(n_factors=nr_factori, rotation=None)
    model_af_0.fit(x)

    l0 = model_af_0.loadings_  # matrice (m variabile x nr_factori)

    t_l0 = pd.DataFrame(
        l0,
        index=variabile_observate,
        columns=etichete_factori
    )
    t_l0.index.name = "Indicator"
    t_l0.to_csv("data_out/data_out_fa/l_fara_rotatie.csv")

    print("\nLoadings fara rotatie:")
    print(t_l0.round(3))

    # --- 2) Model CU rotatie (varimax) ---
    model_af = fa.FactorAnalyzer(n_factors=nr_factori, rotation="varimax")
    model_af.fit(x)

    l = model_af.loadings_

    t_l = pd.DataFrame(
        l,
        index=variabile_observate,
        columns=etichete_factori
    )
    t_l.index.name = "Indicator"
    t_l.to_csv("data_out/data_out_fa/l.csv")

    print("\nLoadings cu rotatie (varimax):")
    print(t_l.round(3))

    # ------------------------------------------------------------------------------
    # B3 (VARIANTA INTERPRETARE ECONOMICA) – 3 FACTORI
    # ------------------------------------------------------------------------------

    nr_factori_econ = 3
    etichete_factori_econ = ["F" + str(i) for i in range(1, nr_factori_econ + 1)]

    # --- 1) Model FARA rotatie (3 factori) ---
    model_af_0_econ = fa.FactorAnalyzer(n_factors=nr_factori_econ, rotation=None)
    model_af_0_econ.fit(x)

    l0_econ = model_af_0_econ.loadings_

    t_l0_econ = pd.DataFrame(
        l0_econ,
        index=variabile_observate,
        columns=etichete_factori_econ
    )
    t_l0_econ.index.name = "Indicator"
    t_l0_econ.to_csv("data_out/data_out_fa/l_fara_rotatie_3f.csv")

    print("\nLoadings fara rotatie – 3 factori (interpretare economica):")
    print(t_l0_econ.round(3))

    # --- 2) Model CU rotatie (varimax) – 3 factori ---
    model_af_econ = fa.FactorAnalyzer(n_factors=nr_factori_econ, rotation="varimax")
    model_af_econ.fit(x)

    l_econ = model_af_econ.loadings_

    t_l_econ = pd.DataFrame(
        l_econ,
        index=variabile_observate,
        columns=etichete_factori_econ
    )
    t_l_econ.index.name = "Indicator"
    t_l_econ.to_csv("data_out/data_out_fa/l_3f.csv")

    print("\nLoadings cu rotatie (varimax) – 3 factori (interpretare economica):")
    print(t_l_econ.round(3))

    # ------------------------------------------------------------------------------
    # B4. SCORURI FACTORIALE (CU si FARA ROTATIE)
    # ------------------------------------------------------------------------------

    # setam indexul observatiilor (tari)
    etichete_obs = list(t.index)

    # =========================
    # B4.1 – 2 FACTORI (Kaiser)
    # =========================
    nr_factori = 2
    etichete_factori = ["F" + str(i) for i in range(1, nr_factori + 1)]

    # (re)fit modele – ca blocul sa fie robust, chiar daca ai rulat partial
    model_af_0 = fa.FactorAnalyzer(n_factors=nr_factori, rotation=None)
    model_af_0.fit(x)

    model_af = fa.FactorAnalyzer(n_factors=nr_factori, rotation="varimax")
    model_af.fit(x)

    # Scoruri fara rotatie
    f0 = model_af_0.transform(x)
    t_f0 = pd.DataFrame(f0, index=etichete_obs, columns=etichete_factori)
    t_f0.index.name = t.index.name  # "Countries"
    t_f0.to_csv("data_out/data_out_fa/f_fara_rotatie.csv")

    print("\nScoruri factoriale fara rotatie (2 factori):")
    print(t_f0.round(3))

    # Scoruri cu rotatie (Varimax)
    f = model_af.transform(x)
    t_f = pd.DataFrame(f, index=etichete_obs, columns=etichete_factori)
    t_f.index.name = t.index.name
    t_f.to_csv("data_out/data_out_fa/f.csv")

    print("\nScoruri factoriale cu rotatie Varimax (2 factori):")
    print(t_f.round(3))

    # =========================================
    # B4.2 – 3 FACTORI (interpretare economica)
    # =========================================
    nr_factori_econ = 3
    etichete_factori_econ = ["F" + str(i) for i in range(1, nr_factori_econ + 1)]

    model_af_0_econ = fa.FactorAnalyzer(n_factors=nr_factori_econ, rotation=None)
    model_af_0_econ.fit(x)

    model_af_econ = fa.FactorAnalyzer(n_factors=nr_factori_econ, rotation="varimax")
    model_af_econ.fit(x)

    # Scoruri fara rotatie – 3 factori
    f0_econ = model_af_0_econ.transform(x)
    t_f0_econ = pd.DataFrame(f0_econ, index=etichete_obs, columns=etichete_factori_econ)
    t_f0_econ.index.name = t.index.name
    t_f0_econ.to_csv("data_out/data_out_fa/f_fara_rotatie_3f.csv")

    print("\nScoruri factoriale fara rotatie (3 factori):")
    print(t_f0_econ.round(3))

    # Scoruri cu rotatie (Varimax) – 3 factori
    f_econ = model_af_econ.transform(x)
    t_f_econ = pd.DataFrame(f_econ, index=etichete_obs, columns=etichete_factori_econ)
    t_f_econ.index.name = t.index.name
    t_f_econ.to_csv("data_out/data_out_fa/f_3f.csv")

    print("\nScoruri factoriale cu rotatie Varimax (3 factori):")
    print(t_f_econ.round(3))

    # ------------------------------------------------------------------------------
    # B5. COMUNALITATI (pe solutia cu rotatie, 3 factori)
    # ------------------------------------------------------------------------------

    model_af_econ = fa.FactorAnalyzer(n_factors=3, rotation="varimax")
    model_af_econ.fit(x)

    comm = model_af_econ.get_communalities()

    t_comm = pd.DataFrame(
        {"Comunalitate": comm},
        index=variabile_observate
    )
    t_comm.index.name = "Indicator"

    t_comm.to_csv("data_out/data_out_fa/Comm.csv")

    print("\nComunalitati:")
    print(t_comm.round(3))


if __name__ == "__main__":
    main()
