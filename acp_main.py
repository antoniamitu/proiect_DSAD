import sys
import numpy as np
import pandas as pd

from acp_functii import nan_replace_df, acp, tabelare_varianta, salvare_ndarray
from acp_grafice import plot_varianta, corelograma, plot_scoruri_corelatii

pd.set_option("display.max_columns",None)
np.set_printoptions(3,sys.maxsize,suppress=True)

t=pd.read_csv("data_in/economicdata2023-2023.csv")
nan_replace_df(t)  # din seminarul 8 – functia nan_replace_df (inlocuire lipsuri)\

#Seminar 8 indentificare coloane relevenate
#avem 2 coloane de identificare: ISO_Code, Countries
variabile_observate=list(t)[2:]

# Setam 'Countries' ca index pentru a avea etichetele corecte pe grafice
t.set_index("Countries", inplace=True)

x=t[variabile_observate].values

#ACP
x_,r_v,alpha,a = acp(x)

# Tabele: distributia variantei
t_varianta = tabelare_varianta(alpha)
t_varianta.round(3).to_csv("data_out/data_out_acp/distributia_variantei.csv")

#Grafice: plot varianta componente
k = plot_varianta(alpha)
nr_componente = min([v for v in k if v is not None])
print("Nr minim componente recomandate:", nr_componente)

# --- FORTARE PENTRU RAPORT ---
nr_componente = 3
print(f"-> Vom folosi {nr_componente} componente (conform deciziei din raport).")

#Tabele: matricea de corelatii intre variabile
t_r = salvare_ndarray(r_v,variabile_observate,variabile_observate,"Indicatori","data_out/data_out_acp/corelatii_variabile.csv")
#Grafice: corelograma
# Modificam 10 in 15 ca sa apara cifrele (fiind 11 variabile)
corelograma(t_r, titlu="corelograma_variabile", annot=len(variabile_observate)<15)

# Calcul componente
c = x_@a
n,m = x.shape

# Tabel: corelațiile dintre variabilele observate și componentele principale
r_xc = np.corrcoef(x_,c,rowvar=False)[:m,m:]
# Selectam doar primele 3 componente (taiem tabelul automat)
r_xc_final = r_xc[:, :nr_componente]

# DEFINIM NUMELE CELOR 3 COLOANE (Fixul critic!)
col_comp = ["C"+str(i+1) for i in range(nr_componente)]

t_r_xc = salvare_ndarray(r_xc_final, variabile_observate, col_comp, "Indicatori", "data_out/data_out_acp/corelatii_variabile_componente.csv")
corelograma(t_r_xc,"corelograma_corelatii_variabile_componente",annot=m<15)

#Grafic:plot corelații dintre variabilele observate și componente (cercul corelațiilor)
for i in range(1,nr_componente):
    for j in range(i+1,nr_componente+1):
        plot_scoruri_corelatii(t_r_xc,
            varx="C"+str(i),
            vary="C"+str(j),
            titlu="cercul_corelatiilor",
            etichete=t_r_xc.index,
            corelatii=True
        )

# Grafice: plot scoruri
s = c/np.sqrt(alpha)
# Selectam scorurile doar pentru cele 3 componente
c_final = c[:, :nr_componente]
s_final = s[:, :nr_componente]


t_c = salvare_ndarray(c_final,t.index,t_r_xc.columns,t.index.name,"data_out/data_out_acp/scoruri_componente.csv")
t_s = salvare_ndarray(s_final,t.index,t_r_xc.columns,t.index.name,"data_out/data_out_acp/scoruri_standardizate_componente.csv")
for i in range(1,nr_componente):
    for j in range(i+1,nr_componente+1):
        plot_scoruri_corelatii(t_c,
                               varx="C"+str(i),
                               vary="C"+str(j),
                               titlu="plot_scoruri_tari",
                               etichete=t.index)



# Tabel: Cosinusurile
c2 = c * c
cosinus_total = (c2.T / np.sum(c2, axis=1)).T

# IMPORTANT: Selectam doar primele 3 coloane pentru a salva in CSV-ul nostru (C1, C2, C3)
cosinus_final = cosinus_total[:, :nr_componente]
t_cosin = salvare_ndarray(
    cosinus_final,
    t.index,
    col_comp,
    t.index.name,
    "data_out/data_out_acp/tabel_cosinusuri.csv"
)

corelograma(t_cosin,"cosinusuri",0,1,"Greens",annot=False)

# Tabel: Contributii
c2_final = c_final * c_final
contributii = c2_final*100/np.sum(c2_final,axis=0)
t_contrib = salvare_ndarray(
    contributii,
    t.index,
    col_comp,
    t.index.name,
    "data_out/data_out_acp/tabel_contributii.csv"
)
corelograma(t_contrib,"contributii",0,5,"Blues",annot=False)

# Tabel: Comunalitati
r2 = r_xc_final*r_xc_final
comunalitati = np.cumsum(r2,axis=1)
t_comm = salvare_ndarray(
    comunalitati,
    variabile_observate,
    col_comp,
    t_r_xc.index.name,
    "data_out/data_out_acp/tabel_comunalitati.csv"
)
corelograma(
    t_comm,
    "comunalitati",
    0,
    cmap="Reds",
    annot=len(variabile_observate)<15
)

