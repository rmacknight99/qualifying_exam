import sys, os
import autode as ade
from autode.input_output import xyz_file_to_atoms
from autode.mol_graphs import make_graph
from autode.calculation import Calculation
import autode.wrappers.keywords as kws
import pandas as pd
import itertools
from rdkit import Chem

# define some global stuff

opt_keywords = ['M062X', 'DEF2-TZVPD', 'Opt', 'RIJCOSX', 'D3zero', 'CPCM']
hess_keywords = ['M062X', 'DEF2-TZVPD', 'Freq', 'RIJCOSX', 'D3zero']
solvent_block = f'%cpcm\nsmd true\nSMDsolvent "ACETONITRILE"\nend'

orca = ade.methods.ORCA()
metals = pd.read_csv("metals.txt")["SYMBOL"].tolist()

# find epoxide and metal

def write_mol_file(xyz_file):

    os.system(f"obabel -ixyz {xyz_file} -omol -O tmp.mol > /dev/null 2>&1")
    mol = Chem.MolFromMolFile("tmp.mol", sanitize=False)
    os.system("rm tmp.mol")

    return mol

def get_indices(mol):

    indices = []
    for i, a in enumerate(mol.GetAtoms()):
        this_atom_symbol = a.GetSymbol()
        neighbor_atoms = [x.GetSymbol() for x in a.GetNeighbors()]
        if this_atom_symbol == "O":
            if neighbor_atoms == ["C", "C"]:
                indices.append(i)
        elif this_atom_symbol in metals:
            indices.append(i)

    return indices

# define calculation functions

def optimize_with_orca(molecule, method, keywords, n_cores, solvent_block=None, constrain=None):

    calc = Calculation(name=f'{molecule.name}_opt',
                       molecule=molecule,
                       method=method,
                       keywords=kws.OptKeywords(keywords),
                       other_input_block=solvent_block,
                       n_cores=n_cores,
                       cartesian_constraints=constrain)

    return calc 

def frequency_with_orca(molecule, method, keywords, n_cores, solvent_block=None, constrain=None):

    calc = Calculation(name=f'{molecule.name}_freq',
                       molecule=molecule,
                       method=method,
                       keywords=kws.SinglePointKeywords(keywords),
                       other_input_block=None,
                       n_cores=n_cores,
                       cartesian_constraints=constrain)

    return calc

def opt_and_thermo(mol, orca_dir, LA, n_cores, constrain=None):

    path = os.getcwd()
    os.chdir(orca_dir)
    os.system(f"mkdir -p {LA}")
    os.chdir(LA)
    print(f"\n\toptimizing {LA}", flush=True)

    err = False
    try:
        err = check_opt()
    except:
        pass
    
    if not err:
        opt = optimize_with_orca(mol, orca, opt_keywords, int(n_cores), solvent_block=solvent_block, constrain=constrain)
        opt.run()
        err = check_opt()
    else:
        print(f"\toptimization failed", flush=True)
        
    if not err:
        print(f"\toptimization successful", flush=True)
        print(f"\tfrequency calculation {LA}", flush=True)
        freq = frequency_with_orca(mol, orca, hess_keywords, int(n_cores), solvent_block=solvent_block, constrain=constrain)
        freq.run()
        free_energy = get_free_energy()
    else:
        free_energy = None
    os.chdir(path)

    return free_energy

def get_free_energy():

    free_energy = None
    for i in os.listdir("."):
        if "crest_freq_orca.out" in i:
            out_file = i

    with open(f"{out_file}", "r") as f:
        for line in f:
            line.strip()
            linelist = line.split()
            if "G-E(el)" in linelist:
                free_energy = float(linelist[-2])

    return free_energy

def check_opt():

    err = False
    for i in os.listdir("."):
        if "crest_opt_orca.out" in i:
            out_file = i
    with open(f"{out_file}", "r") as f:
        for line in f:
            line.strip()
            linelist = line.split()
            if "aborting" in linelist:
                err = True

    return err

def run(sub_dirs, n_cores):

    energies_dict = {}
    for i, e in enumerate(sub_dirs):
        print(f"looking in {e}...", flush=True)
        orca_dir = f"{e}_ORCA/"
        os.system(f"mkdir -p {orca_dir}")
        
        for xyz_file in os.listdir(e):
            LA = xyz_file.split("_")[0] + "_" + xyz_file.split("_")[1]
            if LA.endswith("i"):
                LA += "_"+xyz_file.split("_")[2]
            elif LA.endswith("v"):
                LA += "_"+xyz_file.split("_")[2]
            elif LA.endswith("er"):
                LA = LA.split("_")[0]
            
            if LA in energies_dict:
                pass
            else:
                energies_dict[LA] = [0 for v in sub_dirs]

            os.system(f"cp {e}/{xyz_file} {orca_dir}")
            atoms = xyz_file_to_atoms(f"{orca_dir}{xyz_file}") # get atoms 
            electrons = sum(atom.atomic_number for atom in atoms) # get number of electrons
            charge = 0 # charge should always be zero for Lewis acids
            if electrons % 2 != charge: # if there is an unpaired electron
                unpaired_electrons = 1 
                mult = int(2 * 1/2 * unpaired_electrons + 1) # set multiplicity accordingly
            else:
                mult = 1 # if no unpaired electron

            mol = ade.Molecule(f"{orca_dir}{xyz_file}", charge=charge, mult=mult)
            if "complexes" in e:
                home = os.getcwd()
                os.chdir(f"{orca_dir}")
                rdkit_mol = write_mol_file(f"{xyz_file}")
                constrain = get_indices(rdkit_mol)
                os.chdir(home)
            else:
                constrain = None
            free_energy = opt_and_thermo(mol, orca_dir, LA, int(n_cores), constrain)
            energies_dict[LA][i] = free_energy

    df = pd.DataFrame(energies_dict).round(1)
    df = df.transpose().set_axis(sub_dirs, axis=1, inplace=False)

    return df

def gather(sub_dirs, n_cores, epoxide_energy=None):
    
    df = run(sub_dirs=sub_dirs, n_cores=n_cores)
    
    epoxide_col = [epoxide_energy for i in list(range(df.shape[0]))]
    dimerization_energy = [0.0 for i in list(range(df.shape[0]))]
    m_complexation_energy = [0.0 for i in list(range(df.shape[0]))]
    d_complexation_energy = [0.0 for i in list(range(df.shape[0]))]
    df["styrene_oxide"] = epoxide_col

    if "monomers" in sub_dirs and "dimers" in sub_dirs:
        dimerization_energy = 2*df["dimers"] - df["monomers"]
        dimerization_energy = dimerization_energy.tolist()
        for i, e in enumerate(df["dimers"].tolist()):
            if e == 0: # no dimer form, set to None
                dimerization_energy[i] = 0
            else: # dimer form, round 
                dimerization_energy[i] = round(dimerization_energy[i], 1)
                
    if "monomer_complexes" in sub_dirs and "dimer_complexes" in sub_dirs:
        m_complexation_energy = (df["monomer_complexes"]) - (df["monomers"] + df["styrene_oxide"])
        d_complexation_energy = (df["dimer_complexes"]) - (df["dimers"] + df["styrene_oxide"])                                                       
        for i, e in enumerate(df["monomer_complexes"].tolist()):
            if e == 0: # monomer complex failed to calculate
                m_complexation_energy[i] = 0
            else: # monomer complex, round
                m_complexation_energy[i] = round(m_complexation_energy[i], 2) 
        for i, e in enumerate(df["dimer_complexes"].tolist()):
            if e == 0: # dimer complex failed to calculate
                d_complexation_energy[i] = 0
            else: # dimer complex, round
                d_complexation_energy[i] = round(d_complexation_energy[i], 2)
                
    df["monomer_complexation_energy"] = [round(v, 1) for v in m_complexation_energy]            
    df["dimerization_energy"] = [round(v, 1) for v in dimerization_energy]
    df["dimer_complexation_energy"] = [round(v,1) for v in d_complexation_energy]
    
    return df.sort_index()

def best_complex(df):

    df = df.drop(["dimers", "monomers", "monomer_complexes", "dimer_complexes", "styrene_oxide"], axis=1)
    print(df, flush=True)
        
