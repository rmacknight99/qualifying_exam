import pandas as pd
import json, os

summary = pd.read_csv("summary_09_28_2022.csv", index_col=0)
columns = list(summary.columns)
energies = summary.T.to_dict('list')
best_complexes = {key:"" for key in energies.keys()}
best_energies = {key:float(0.0) for key in energies.keys()}
for key, value in energies.items():
    if value[0] != value[0]:
        #print(f"{key} has no dimer form\n")
        best_complexes[key] = "monomer_complex"
        best_energies[key] = round(energies[key][1], 3)
    elif value[0] >= 0:
        #print(f"{key} has positive dimerization energy\n")
        best_complexes[key] = "monomer_complex"
        best_energies[key] = round(energies[key][1], 3)
    elif value[0] < 0:
        if value[2] >= 0:
            best_complexes[key] = "monomer_complex"
            best_energies[key] = round(energies[key][1], 3)
        else:
            best_complexes[key] = "dimer_complex"
            best_energies[key] = round(energies[key][2], 3)


os.system("mkdir -p best_complexes/")
for key, value in best_complexes.items():
    d = value + "es" + "/orca/"
    filename = key + "_" + value.split("_")[0] + "_crest_opt_orca.xyz"
    os.system(f"cp {d+filename} best_complexes/{key}_complex.xyz")

with open("best_complexes.json", "w") as f:
    json.dump(best_complexes, f, indent=4)

with open("best_energies.json", "w") as f:
    json.dump(best_energies, f, indent=4)

