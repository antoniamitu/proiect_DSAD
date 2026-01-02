import os
import pandas as pd
import matplotlib.pyplot as plt

from fa_grafice_utilitar import corelograma, plot_scoruri_corelatii, show, harta_scoruri_factoriale

os.makedirs("graphics/graphics_fa", exist_ok=True)
plt.rcParams.update({"font.size": 8})

# abrevieri scurte pentru variabile (ca să nu se suprapună)
abbr = [
    "GovCons", "TransfSub", "JudInd", "PropRights", "Infl_SD",
    "Infl", "NT_Barr", "CapCtrl", "CredMkt", "HireMinW", "BusReg"
]

# C1 KMO
t_kmo = pd.read_csv("data_out/data_out_fa/kmo.csv", index_col=0)
corelograma(t_kmo, "Index KMO", 0, cmap="Reds")

# ========== 2 factori ==========
t_l0_2 = pd.read_csv("data_out/data_out_fa/l_fara_rotatie.csv", index_col=0)
corelograma(t_l0_2, "Corelograma AFact fara rotatie (2 factori)")
plot_scoruri_corelatii(t_l0_2, "F1", "F2", "Plot corelatii AFact fara rotatie (2 factori)", abbr, corelatii=True)

t_l_2 = pd.read_csv("data_out/data_out_fa/l.csv", index_col=0)
corelograma(t_l_2, "Corelograma AFact (Varimax, 2 factori)")
plot_scoruri_corelatii(t_l_2, "F1", "F2", "Plot corelatii AFact (Varimax, 2 factori)", abbr, corelatii=True)

# scoruri (NU etichetăm țările)
t_f0_2 = pd.read_csv("data_out/data_out_fa/f_fara_rotatie.csv", index_col=0)
plot_scoruri_corelatii(t_f0_2, "F1", "F2", "Plot scoruri fara rotatie (2 factori)", etichete=None)

t_f_2 = pd.read_csv("data_out/data_out_fa/f.csv", index_col=0)
plot_scoruri_corelatii(t_f_2, "F1", "F2", "Plot scoruri (Varimax, 2 factori)", etichete=None)

# ========== 3 factori (interpretare) ==========
t_l0_3 = pd.read_csv("data_out/data_out_fa/l_fara_rotatie_3f.csv", index_col=0)
corelograma(t_l0_3, "Corelograma AFact fara rotatie (3 factori)")

t_l_3 = pd.read_csv("data_out/data_out_fa/l_3f.csv", index_col=0)
corelograma(t_l_3, "Corelograma AFact (Varimax, 3 factori)")
plot_scoruri_corelatii(t_l_3, "F1", "F2", "Plot corelatii AFact (Varimax, 3 factori) F1-F2", abbr, corelatii=True)
plot_scoruri_corelatii(t_l_3, "F1", "F3", "Plot corelatii AFact (Varimax, 3 factori) F1-F3", abbr, corelatii=True)

t_f_3 = pd.read_csv("data_out/data_out_fa/f_3f.csv", index_col=0)
plot_scoruri_corelatii(t_f_3, "F1", "F2", "Plot scoruri (Varimax, 3 factori) F1-F2", etichete=None)
plot_scoruri_corelatii(t_f_3, "F1", "F3", "Plot scoruri (Varimax, 3 factori) F1-F3", etichete=None)

# C8 comunalitati
t_comm = pd.read_csv("data_out/data_out_fa/Comm.csv", index_col=0)
corelograma(t_comm, "Comunalitati", 0, cmap="Blues")

t_initial = pd.read_csv("data_in/economicdata2023-2023.csv")
t_initial.set_index("Countries", inplace=True)

# Harta scoruri (Varimax, 2 factori)
harta_scoruri_factoriale(t_f_2, t_initial, coloana_iso="ISO_Code",
                         titlu="harta_scoruri_factoriale_varimax_2f")

# Harta scoruri (Varimax, 3 factori)
harta_scoruri_factoriale(t_f_3, t_initial, coloana_iso="ISO_Code",
                         titlu="harta_scoruri_factoriale_varimax_3f")

#show()
