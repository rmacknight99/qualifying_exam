from morfeus import *
import os, json
import pandas as pd
import numpy as np
from src.descriptors.models import FeatureImportance as FI
from sklearn.preprocessing import RobustScaler

class Descriptors(object):

    def __init__(self, home=os.getcwd(), geom_dir="xyz_files", metals="metals.txt", gen_hess=False,
                 FI_path="data_folder/FI_24_hr.csv", FI_obj="sn2", desc_loc=None, keys_loc=None):

        self.home = home
        self.geom_dir = geom_dir
        self.metals = metals
        self.gen_hess = gen_hess
        self.FI_path = FI_path
        self.FI_obj = FI_obj
        self.desc_loc = desc_loc
        self.keys_loc = keys_loc
        
        self.bond_dicts = []
        
    def xtb_hess(self):

        os.system("mkdir -p calcs/") # make calcs folder
        os.system("mkdir -p hessian_files/") # make hessian files
        cwd = os.getcwd()
        for xyz_file in os.listdir("."): # iterate over xyz files
            if "xyz" in xyz_file:
                name = xyz_file[:-4]
                os.system(f"cp {xyz_file} calcs/")
                os.chdir("calcs")
                os.system(f"xtb {xyz_file} --hess")
                os.chdir(cwd)
                dest = f"hessian_files/{name}"
                os.system(f"mkdir -p {dest}")
                os.system(f"cp calcs/hessian {dest}")

    def find_epoxide(self):
        """
        Find index of components in epoxide ring for a given Epoxide+LA complex
        """
        LFC_dict = {}
        metals = self.metal_list
        # get bond information from internal coordinates
        for index, bond in enumerate(self.int_coords):
            idx_1 = bond.i - 1
            idx_2 = bond.j - 1
            key = (self.elements[idx_1] + "_" + str(idx_1), self.elements[idx_2] + "_" + str(idx_2))
            if self.LFCs is not None:
                LFC_dict[key] = self.LFCs[index]
        carbon_oxygen = []
        carbon_carbon = []
        metal_oxygen = []
        for tup in LFC_dict.keys(): # iterate over bonds
            if "C_" in tup[0] or "C_" in tup[1]: # if carbon is involved
                if "O_" in tup[0] or "O_" in tup[1]: # if oxygen is involved
                    carbon_oxygen.append(tup) # add to list of C-O bonds
                elif "C_" in tup[0] and "C_" in tup[1]: # if carbon is involved
                    carbon_carbon.append(tup) # add to list of C-C bonds
            elif tup[0].split("_")[0] in metals or tup[1].split("_")[0]: # if there is a metal
                if "O_" in tup[0] or "O_" in tup[1]: # if there is an oxygen
                    metal_oxygen.append(tup) # add to list of M-O bonds

        carbon_indices = []
        index_dict = {}
        if len(carbon_oxygen) == 2: # if we only found two C-O bonds we are done
            for b in carbon_oxygen:
                oxygen_index = b[0].split("_")[1]
                carbon_indices.append(b[1].split("_")[1])
            index_dict["O"] = int(oxygen_index)
            index_dict["C"] = [int(element) for element in carbon_indices]
        else: # do some more digging
            carbons = []
            for b in carbon_oxygen:
                for atom in b:
                    if "C" in atom:
                        carbons.append(atom)
            for b in carbon_carbon:
                if b[0] in carbons and b[1] in carbons:
                    carbon_indices.append(b[0].split("_")[1])
                    carbon_indices.append(b[1].split("_")[1])
            index_dict["C"] = [int(element) for element in carbon_indices]
            for b in carbon_oxygen:
                if b[0].split("_")[1] in carbon_indices or b[1].split("_")[1] in carbon_indices:
                    if "O" in b[0]:
                        oxygen_index = b[0].split("_")[1]
                    else:
                        oxygen_index = b[1].split("_")[1]
            index_dict["O"] = int(oxygen_index)
    
        for i in [0,1]: # label C-O bonds based on number of hydrogens bonded to C
            Hs = 0
            for tup in LFC_dict.keys():
                if "C_"+str(index_dict["C"][i]) in tup:
                    if "H_" in tup[0] or "H_" in tup[1]:
                        Hs +=1
            if Hs > 1:
                index_dict["C_1H"] = index_dict["C"][i]
            else:
                index_dict["C_2H"] = index_dict["C"][i]

        bond_1 = [index_dict["O"]+1, index_dict["C_1H"]+1]
        bond_2 = [index_dict["O"]+1, index_dict["C_2H"]+1]
        bond_3 = [index_dict["C_1H"]+1, index_dict["C_2H"]+1]
        bond_dict = {"oxygen_to_CH": bond_1, "oxygen_to_CH2": bond_2, "carbon_carbon": bond_3}

        self.bond_dicts.append(bond_dict)
        self.bond_dict = bond_dict

    def get_descriptors(self, exclude=None):

        print("\ngenerating descriptors...\n")
        os.chdir(self.geom_dir) # enter directory with XYZ files
        formula_to_name = json.load(open("formula_to_name.json", "r")) # load dictionary to get LA name from LA molecular formula
        self.metal_list = pd.read_csv(self.metals)["SYMBOL"].tolist() # get list of metals
        descriptors = {} # initialize descriptors
        
        if self.gen_hess == True:
            self.xtb_hess() # generate hessian files with xTB

        for i in os.listdir("."): # iterate over xyz files
            if ".xyz" in i:
                filename = i
                # morfeus calculations
                print("\n\tgathering MORFEUS descriptors...")
                elements, coordinates = read_xyz(i)
                # solvent accesisible surface area
                sasa = SASA(elements, coordinates)
                sasa_area = sasa.area
                sasa_volume = sasa.volume
                # dispersion descriptors                                                                                                              
                disp = Dispersion(elements, coordinates)
                disp.compute_coefficients()
                disp.compute_p_int()
                disp_area = disp.area
                disp_volume = disp.volume
                disp_p_int = disp.p_int
                disp_p_max = disp.p_max
                disp_p_min = disp.p_min
                potential_descs = {"sasa_area":sasa_area, "sasa_volume":sasa_volume, 
                                   "disp_area":disp_area, "disp_volume":disp_volume, 
                                   "disp_p_int":disp_p_int, "disp_p_max":disp_p_max,
                                   "disp_p_min":disp_p_min}
                # xtb
                try:
                    xtb = XTB(elements, coordinates)
                    print("\tgathering xTB descriptors...")
                    ip = xtb.get_ip()
                    ea = xtb.get_ea()                                                                                                               
                    electrophilicity = xtb.get_global_descriptor("electrophilicity", corrected=True)
                    nucleophilicity = xtb.get_global_descriptor("nucleophilicity", corrected=True)
                    potential_descs["ip"] = ip
                    potential_descs["ea"] = ea
                    potential_descs["electrophilicity"] = electrophilicity
                    potential_descs["nucleophilicity"] = nucleophilicity
                except:
                    print("\tfailed to gather xTB descriptors...")
                    pass

                # local force constants and frequencies
                lf = LocalForce(elements, coordinates)
                try:
                    path = f"hessian_files/{filename[:-4]}/hessian"
                    print("\tgathering LFCs...")
                    lf.load_file(path, "xtb", "hessian")
                    lf.normal_mode_analysis()
                    lf.detect_bonds()
                    lf.compute_local()
                    lf.compute_frequencies()
                    lf.compute_compliance()
                    local_force_constants = lf.local_force_constants
                    local_frequencies = lf.local_frequencies
                except:
                    local_force_constants = None
                    local_frequencies = None
            # get local force constants for epoxide ring
            if local_force_constants is not None:
                self.elements = elements
                self.int_coords = lf.internal_coordinates
                self.LFCs = local_force_constants
                self.find_epoxide()
                bonds = list(self.bond_dict.values())
                bond_names = list(self.bond_dict.keys())
                for i, bond in enumerate(bonds):
                    potential_descs[bond_names[i]] = lf.get_local_force_constant(bond)
            else:
                print(f"\tcould not get local force constants for {filename}")
            # add to descriptors dictionary key(molecular formula):value(descriptor dictionary)
            for key in formula_to_name.keys():
                if filename.startswith(key):
                    descriptors[formula_to_name[key]] = list(potential_descs.values())

        os.chdir(self.home) # go back to home
        print(f"\nDescriptors Calculated — " + ', '.join([key for key in potential_descs.keys()]))
        keys = list(potential_descs.keys())

        if exclude is not None:
            exclude = [i for i in exclude if i in descriptors.keys()]
            print(f"excluding {', '.join(exclude)}")
            descriptors = {key: descriptors[key] for key in descriptors if key not in exclude}

        self.descriptors = descriptors
        self.keys = keys
        json.dump(self.descriptors, open("desc_folder/all_lewis_acid_descs.json", "w"))
                
    def load_training(self):

        all_df = pd.read_csv(self.FI_path)
        LA = all_df["lewis_acid"]
        obj = all_df[self.FI_obj]

        self.LA = LA
        self.obj = obj
        
    def sub_descriptors(self):

        self.load_training()
        described_LAs = []
        described_objs = []
        for i, e in enumerate(self.LA):
            try:
                described_LAs.append(self.descriptors[e])
                described_objs.append(self.obj[i])
                print(f"\t{e} has descriptors")
            except:
                print(f"\t{e} has no descriptors")
                described_LAs.append([pd.NA for key in self.keys])
                described_objs.append(self.obj[i])
        df = pd.DataFrame(described_LAs, columns=self.keys)
        df[self.FI_obj] = described_objs
        df.index = self.LA
        print(f"\n{df.shape[0]} observations for feature importance evaluation\n")
        df = df.dropna(axis=0, how="any")
        self.training_df = df

    def dump_descriptors(self):

        json.dump(self.descriptors, open(self.desc_loc, "w"))
        with open(self.keys_loc, "w") as f:
            for key in self.keys:
                f.write(key+"\n")
    
        print(f"\ndescriptors dumped to {self.desc_loc} and keys written to {self.keys_loc}")

    def load_descriptors(self, exclude=None):

        self.descriptors = json.load(open(self.desc_loc, "rb"))
        self.keys = []
        with open(self.keys_loc, "r") as f:
            for line in f:
                line.strip()
                linelist = line.split()
                self.keys.append(linelist[0])

        if exclude is not None:
            exclude = [i for i in exclude if i in descriptors.keys()]
            self.descriptors = {key: self.descriptors[key] for key in self.descriptors if key not in exclude}
            
        print(f"\ndescriptors loaded from {self.desc_loc} and keys loaded from {self.keys_loc}")
        
    def get_feature_importance(self, model_type="RF", n_models=10):

        scaler = RobustScaler()
        cols = list(self.training_df.columns)[:-1]
        self.training_df[cols] = scaler.fit_transform(self.training_df[cols])
        features = self.training_df.iloc[:,:-1]
        y = self.training_df.iloc[:,-1:]
        models = ["Linear", "RF", "BNN"]

        fi = FI(features, y, features_to_keep=5)
        self.model_type = model_type
        importances_dict = {key: 0 for key in list(features.columns)}
        if self.model_type == "Linear":
            pass
        if self.model_type == "RF":
            feature_names = list(features.columns)
            track_imp_feats = dict.fromkeys(feature_names, 0)
            train_scores = np.zeros(n_models)
            for i in range(n_models):
                imp_feats, train_score, fis = fi.RFModel(100)
                for j, k in enumerate(list(importances_dict.keys())):
                    importances_dict[k] += fis[j]
                for f in imp_feats:
                    track_imp_feats[f] += 1
                train_scores[i] = round(train_score, 3)
            keys = np.array(list(track_imp_feats.keys()))
            values = np.array(list(track_imp_feats.values()))
            opt_feats = values.argsort()[-5:]
        if self.model_type == "BNN":
            pass

        if self.model_type not in models:
            print(f"model type `{model_type}` not found")
        
        self.importances_dict = {key: value/n_models for key, value in list(importances_dict.items())}
        self.opt_feats = opt_feats
        self.train_scores = train_scores
        
    def trim_descriptors(self, model_type="RF", n_models=10):

        self.get_feature_importance(model_type = model_type, n_models = n_models)

        for k, v in self.descriptors.items():
            trim_v = [v[index] for index in self.opt_feats]
            self.descriptors[k] = trim_v
            
    def opt_format(self):

        # check current format of descriptors
        vals = list(self.descriptors.values())
        # assign keys 
        try: # reformate and scale
            self.descriptor_keys = list(vals[0].keys())
            scale = True
            for k, v in self.descriptors.items():
                trim_v = [v[key] for key in self.descriptor_keys if key in v]
                self.descriptors[k] = trim_v
        except: # dont scale
            scale = False
            self.descriptor_keys = []
            with open(f"{self.desc_loc.split('/')[0]}/curr_desc_keys.txt", "r") as f:
                for line in f:
                    line.strip()
                    ll = line.split()
                    self.descriptor_keys.append(ll[0])
            try:  
                self.descriptor_keys = [self.descriptor_keys[i] for i in self.opt_feats]
            except:
                pass
        
        # make dataframe from descriptors to scale and remove nan
        df = pd.DataFrame.from_dict(self.descriptors, columns = self.descriptor_keys, orient = "index").dropna(axis=0)
        if scale:
            scaler = RobustScaler()
            df[df.columns] = scaler.fit_transform(df[df.columns])
        df = df.round(3)
        # regain descriptors in correct format for optimization and dump
        self.descriptors = df.T.to_dict(orient = "list")
        n_lewis_acids = len(list(self.descriptors.keys()))
        n_descriptors = len(list(self.descriptors.values())[0])
        print(f"Number of Lewis acids profiled: {n_lewis_acids}, Descriptors used: {n_descriptors}\n")

        with open(f"{self.desc_loc.split('/')[0]}/curr_desc_keys.txt", "w") as f:
            for i in self.descriptor_keys:
                f.write(i+"\n")
                
        json.dump(self.descriptors, open(self.desc_loc, "w"))
        
