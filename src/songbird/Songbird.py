import pprint, shutil, os, json, math, pickle, sys, random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
from gryffin import Gryffin
from category_writer import CategoryWriter
from constraints import none_is_zero
from olympus import Olympus
from olympus import list_planners
from olympus import Plotter
from olympus import Dataset
from olympus import ParameterSpace, Parameter
from olympus import Emulator, Campaign, Planner
from olympus.models import BayesNeuralNet
from olympus.objects import ParameterCategorical
from olympus.emulators.emulator import load_emulator
from olympus.utils.data_transformer import DataTransformer
from olympus.utils.misc import r2_score
pp = pprint.PrettyPrinter(indent=5, width=80, compact=True)

class Songbird(object):

    # define optimizer attributes #
    
    def __init__(self, data_path="", num_of_objs=None, config_file=None,
                 parameters=[], objective=[], goal=[], vparams={}, output = None,
                 batched=False, n_cpus=1, auto_desc=False, desc_loc=None, desc_key=None):

        self.vparams = {"parallel": "True", "boosted": "False", "ss schedule": [1, 1],
                        "num_batches": 1, "num_random_samples": 100, "budget": 1, "campaigns": 100, "random": None,
                        "infer": True, "make_categorical": "no", "increment": None, "tolerances": [],
                        "absolute": [], "transform": None, "obj constraint": None, "cut": None,
                        "desc names": [], "normalize descs": "False", "cat details": {}, "cat names": [], "cat descs": [],
                        "log interval": 1, "log": "True", "param constraint": None}

        self.output = output 
        for key in vparams.keys():
            self.vparams[key] = vparams[key]
        
        self.config_file = None 
        if config_file is not None:
            self.parameters = []
            self.objective = []
            self.config_file = config_file
            config_dict = self.load_config()
            for i in self.parameter_list:
                self.parameters.append(i["name"])
            for i in self.objective_list:
                self.objective.append(i["name"])
        else:
            self.data_path = data_path
            df = pd.read_csv(self.data_path)
            columns = list(df.columns)
            parameters = columns[0:-1*int(num_of_objs)]
            objective = columns[-1*int(num_of_objs):]
            self.parameters = list(parameters)
            self.objective = list(objective)
        self.goal = list(goal)
        self.sizes = [1 for i in range(len(self.parameters))]

        self.progress = {}
        for i in self.objective:
            self.progress[i] = []
        self.progress["campaign"] = []
        self.progress["number of observations loaded"] = []
        
        self.batched = batched
        self.n_cpus = n_cpus
        self.auto_desc = auto_desc
        self.desc_key = desc_key
        self.desc_loc = desc_loc
        self.increment = self.vparams["increment"]
        self.num_of_objs = int(num_of_objs)

        self.load()
        general_dict = self.general()
        parameter_list = self.get_parameters()
        objective_list = self.objectives()
        
    def load_config(self):
        with open(self.config_file, "r") as f:
            config_dict = json.load(f)

        self.param_types = []
        for i in config_dict["parameters"]:
            self.param_types.append(i["type"])
        self.config_dict = config_dict
        self.general_dict = config_dict["general"]
        self.parameter_list = config_dict["parameters"]
        self.objective_list = config_dict["objectives"]

        os.rename(self.config_file, f"{self.output}/config.json")
        return config_dict

# to see optimization settings that can be changed #
    
    def help_me(self):

        print("settings:\n")
        for i in self.descriptions.items():
            print(f"{i[0]}: {i[1]}\n")

# load in data from csv file and set optimizer attributes #
    
    def load(self, verbose = True):

            constraint = self.vparams["obj constraint"] # get constraint setting on objective
            
            df = pd.read_csv(self.data_path) # load data
            if self.vparams["transform"] == 'MinMax': # do MinMax scaling if specified
                for i in self.objective:
                    df[i] = MinMaxScaler().fit_transform(np.array(df[i]).reshape(-1,1))                                                                                                               
                    df[i] = [round(j, 4) for j in df[i].tolist()]
                    
            observations = []
            first_row = df.iloc[0].values
            self.param_types = []
            for i in range(len(self.parameters)): # find parameter types
                try:
                    float(first_row[i])
                    self.param_types.append('continuous')
                except:
                    self.param_types.append('categorical')

            lookup_table = []
            for index, r in df.iterrows(): # iterate through dataframe entries
                s = {}
                for key in df.columns[0:len(self.parameters+self.objective)]:
                    s[key] = r[key] # gather sample from dataframe
                lookup_table.append(s) # append dictionary sample to lookup_table
                if constraint is not None: 
                    lower_obj_limit = df.quantile(constraint[0])
                    upper_obj_limit = df.quantile(constraint[1])
                else:
                    lower_obj_limit = [-10000 for i in self.objective]
                    upper_obj_limit = [10000 for i in self.objective]
                fail = False
                for obj in self.objective: # check objective values if they are within constrained region
                    try:
                        if r[obj] < lower_obj_limit[obj] or r[obj] > upper_obj_limit[obj]:
                            fail = True # if outside of range set fail to True
                        else:
                            pass
                    except:
                        pass
                if not fail: # if all objective values are in constrained region append to observations
                    observations.append(s)                    

            if len(lookup_table) == len(observations):
                if verbose:
                    print(f"All {len(lookup_table)} observations loaded")
            else:
                if verbose:
                    print(f"Lookup Table Size => {len(lookup_table)}\n")
                    print(f"Constrained observations loaded => {len(observations)}")
            
            # add attributes 
            self.observations = observations
            self.lookup_table = lookup_table
            self.low_limit = lower_obj_limit
            self.high_limit = upper_obj_limit


    # create general section of dictionary from optimization attributes #
    
    def general(self):

        # set attributes
        parallel = self.vparams["parallel"]
        boosted = self.vparams["boosted"]
        num_batches = self.vparams["num_batches"]

        general_dict = {"num_cpus": self.n_cpus,
                        "parallel": parallel,
                        "boosted": boosted,
                        "verbosity": 3}
        if self.batched == True:
            general_dict["batches"] = num_batches
        if self.auto_desc == True:
            general_dict["auto_desc_gen"] = True
        general_dict["num_random_samples"] = self.vparams["num_random_samples"]
        self.general_dict = general_dict

        return general_dict


    
    # obtain parameter information #
    
    def get_parameters(self, bounds=None):

        parameter_list = []
        df = pd.DataFrame(self.lookup_table)
        cont = 0
        for i,e in enumerate(self.parameters): # add parameters to parameter list                                                            
            # add continuous parameter                                                                               
            if self.param_types[i] == "continuous":
                if self.vparams["infer"] == True:
                    lst = [row[e] for index, row in df.iterrows()]
                    low = float(min(lst))
                    high = float(max(lst))
                else:
                    low = self.vparams["infer"][cont][0]
                    high = self.vparams["infer"][cont][1]
                if self.vparams["make_categorical"] == "yes":
                    param = {"name": e, "type": "categorical",                                                                              
                             "size": 1,
                             "category_details": f"{self.output}/CatDetails/cat_details_{e}.pkl"}
                    self.param_types[i] = "categorical"
                else:
                    param = {"name": e, "type": "continuous",
                             "low": low, "high": high,
                             "size": self.sizes[i]}
                cont += 1
            # add categorical parameter
            else:                                                                                                                             
                param = {"name": e, "type": "categorical",
                         "size": 1,
                         "category_details": f"{self.output}/CatDetails/cat_details_{e}.pkl"}

            parameter_list.append(param) # add parameter to list                                                                              
        self.parameter_list = parameter_list

        return parameter_list


    # obtain objective information #
    
    def objectives(self):

        objective_list = []
        if len(self.objective) > 1: # for multiple objectives
            for i,e in enumerate(self.objective):                                                                                            
                objective_list.append({"name": e, "goal": self.goal[i],                                                  
                                       "hierarchy": i, "tolerance": self.vparams["tolerances"][i],
                                       "absolute": True})
        else:
            objective_list = [{"name": self.objective[0], "goal": self.goal[0]}] # set objective
        self.objective_list = objective_list

        return objective_list


class Optimizer(Songbird):

    # initialize a class from the parent class (Songbird) and add some new methods
    
    def __init__(self, data_path="", num_of_objs=None, config_file=None,
                 parameters=[], objective=[], goal=[], vparams={}, batched=False,
                 n_cpus=1, auto_desc=False, desc_loc=None, desc_key=None, output="archive_optimizer"):

        super().__init__(data_path=data_path, num_of_objs=num_of_objs, config_file=config_file,
                         parameters=parameters, objective=objective, goal=goal, vparams=vparams, output=output,
                         batched=batched, n_cpus=n_cpus, auto_desc=auto_desc, desc_loc=desc_loc, desc_key=desc_key)

        self.vparams["gryffin output"] = output
        try:
            shutil.rmtree(f"{output}")
        except:
            pass
        self.n_seen = []
        os.mkdir(output)

    # write config file/dictionary #
    
    def config(self):
        general_dict = self.general()                                                                                          
        parameter_list = self.get_parameters()                                                        
        objective_list = self.objectives()
        config_dict = {"general":general_dict,                                                                                                    
                       "parameters":parameter_list,                                                                                                
                       "objectives":objective_list} # create config dictionary
        with open(f"{self.vparams['gryffin output']}/config.json", "w") as f:
            json.dump(config_dict, f, indent=4) # write config file

        self.config_dict = config_dict
        return config_dict                                


    # define random sampler (needs work for multi-objective #
    
    def random_sampler(self):

        self.load()
        df = pd.DataFrame(self.lookup_table)
        objs = []
        for i in range(self.vparams["campaigns"]+4):
            s = df.sample(n=1, replace=False)
            objs.append(list(s[self.objective[0]])[0][0])
        if self.goal == 'max':
            objs = [0] + objs
        else:
            objs = [1] + objs
        campaigns = list(range(self.vparams["campaigns"]+5))
        with open("random.out", "w") as f:
            f.write("==== objective ====\n")
            for i,obj in enumerate(objs):
                f.write(f"{campaigns[i]},{obj}\n")

    # write category details #                
    
    def write_categories(self):

        cat_num = 0
        discrete_options = {}
        for i,e in enumerate(self.param_types): # loop over parameter types
            if e == 'categorical': # if we have a categorical parameter
                options = set() # options are a set (no duplicates)
                parameter = self.parameters[i] 
                try:
                    options = list(self.vparams["cat descs"][parameter].keys()) # gather from descriptor dicts
                except:
                    for obs in self.lookup_table:
                        options.add(obs[parameter]) # gather options from existing observations
                try:
                    check_for_vals = self.vparams["cat descs"][parameter].values()
                except:
                    descriptors = {}
                    for index, option in enumerate(options):
                        try:
                            descriptors[option] = [float(option)]
                        except:
                            one_hot = [0] * len(options)
                            one_hot[index] = 1
                            descriptors[option] = one_hot
                    self.vparams["cat descs"][parameter] = descriptors

                category_writer = CategoryWriter(num_opts = len(options), num_dims = len(self.parameters)) # initialize category writer

                descriptors = self.vparams["cat descs"][parameter]
                home_dir = f"{self.vparams['gryffin output']}/"
                p = [parameter]
                if parameter in self.vparams["cat descs"]:
                    print(f"\tgenerating category with descriptors for {parameter} => {self.desc_loc+parameter+'.json'}")
                else:
                    print(f"\tgenerating category with descriptors for {parameter} => one hot or discretized")


                # write cateogry with descriptors
                options = [str(i) for i in options]
                category_writer.write_categories(p, options=options, home_dir=home_dir, descriptors=descriptors, with_descriptors=True)
                cat_num += 1

                
    # extract sample from optimizer #
    
    def get_sample(self, model_path=None):

        for i, sample in enumerate(self.samples):
            found = False
            for entry in self.lookup_table:
                parameters = {}
                for item in list(entry.items())[:len(self.objective_list)+1]:
                    parameters[item[0]] = item[1]
                if parameters == sample:
                    found = True
                    for item in list(entry.items())[-len(self.objective_list):]:
                        sample[item[0]] = round(item[1], 4)
                        self.suggestions.append(sample)
                    else:
                        pass
                    
            for item in list(sample.items()):
                try:
                    sample[item[0]] = round(item[1], 5)
                except:
                    pass
            if not found:
                print("\n\tcould not find observation in lookup table")
                if model_path is not None:
                    print(f"using simulator {model_path} to label...")
                    loaded_em = load_emulator(f"{model_path}")
                    vals = list(sample.values())
                    obj = loaded_em.run([vals])[0][0]
                    sample["peak_area"] = obj
                self.suggestions.append(sample)
            
            self.samples[i] = sample


    # get mode for ss schedule #
            
    def get_mode(self):

        lst = self.vparams["ss schedule"]
        mode = None
        if len(lst) == 2:
            mode = 'one'
        if len(lst) == 4:
            mode = 'multiple'

        return mode

    
    # get sampling strategy schedule #

    def get_ss_schedule(self): 

        schedule = []
        ss_schedule = self.vparams["ss schedule"]
        campaigns = self.vparams["campaigns"]
        mode = self.get_mode()
        
        # for each recommendation one sampling strategie is used (one sample generated)
        if mode == 'one': # [start ss, end ss]
            start = ss_schedule[0]
            end = ss_schedule[1]
            # start and end are not string
            if not isinstance(start, str) and not isinstance(end, str):
                schedule = np.linspace(start, end, campaigns).tolist()
            # start and end are string (always start from explore)
            elif isinstance(start, str) and isinstance(end, str):
                schedule = np.linspace(-1, 1, campaigns).tolist()
            else:
                if isinstance(start, str):
                    schedule = np.linspace(-1, end, campaigns).tolist()
                elif isinstance(end, str):
                    schedule = np.linspace(start, 1, campaigns).tolist()
                    
        # for each recommendation multiple strategies are used (multiple samples generated)
        elif mode == 'multiple': # [start number of strategies, multiplier, increment, max ss]
            cn = 0
            strategies = ss_schedule[0]
            lst = np.linspace(-1, 1, strategies).tolist()
            while cn < campaigns:
                if cn % ss_schedule[2] == 0:
                    strategies = strategies * ss_schedule[1]
                    if strategies <= ss_schedule[-1]:
                        lst = np.linspace(-1, 1, strategies).tolist()
                    else:
                        lst = [1]
                schedule.append(lst)
                cn += 1
            for i, e in enumerate(schedule):
                dummy_lst = []
                for val in e:
                    dummy_lst.append(round(val, 4))
                schedule[i] = dummy_lst
        else:
            print('mode undetermined must be "multiple" or "one"')

        self.ss_schedule = schedule

        return schedule

    # optimziation loop
    
    def optimize(self, obs_log = None, ss = None, model_path = None):

        self.config()
        self.get_ss_schedule()
        budget = self.vparams["budget"]

        if self.vparams["cat names"] != []:
            f_dicts = self.load_descs()
            self.vparams["cat descs"] = f_dicts
        self.write_categories()
        # this block of code removes any observations for which we do not have descriptors for
        for i in f_dicts:
            observed = [entry[i] for entry in self.observations]
            keys = list(f_dicts[i].keys())
            try:
                keys = [float(k) for k in keys]
            except:
                pass
            remove = [index for index, x in enumerate(observed) if x not in keys]
            self.observations = [o for index, o in enumerate(self.observations) if index not in remove]
        ###############################################################################################
        if self.vparams["param constraint"] is not None:
            gryffin = Gryffin(f"{self.vparams['gryffin output']}/config.json", known_constraints=none_is_zero)
        else:
            gryffin = Gryffin(f"{self.vparams['gryffin output']}/config.json")
        print("\nGRYFFIN initialized\n")

        for p in gryffin.config.parameters:
            if p["type"] == "discrete":
                p['specifics']['options'] = self.discrete_options[p['name']]
                descriptors = list(range(len(p['specifics']['options'])))
                p['specifics']['descriptors'] = np.reshape(descriptors, (len(descriptors), 1))
                print(f"{p['name']} has options -> {sorted(p['specifics']['options'])}")
                print(f"{p['name']} has options -> {sorted(p['specifics']['descriptors'])}")
            elif p["type"] == "categorical":
                print(f"{p['name']} has options -> {sorted(p['specifics']['options'])}")
            elif p["type"] == "continuous":
                print(f"{p['name']} has range -> [{p['specifics']['low']}, {p['specifics']['high']}]")

        o = []
        for obj in gryffin.config.objectives:
            o.append(obj['name'])
        print(f"objectives -> {o}")
        
        self.obs_log = []
        self.suggestions = []

        # gather/remove observations for optimizer to load #
        if obs_log is not None:
            self.obs_log = obs_log
        elif self.vparams["cut"] == None:
            try:
                self.obs_log = self.observations
            except:
                self.lookup_table = []
                print("no data available")
        else:
            self.obs_log = random.choices(self.observations, k=self.vparams["cut"])
        
        for self.cn in range(self.vparams["campaigns"]):

            continuous = [] # list of suggested continuous params
            categorical = [] # list of suggested categorical params
                
            # suggestion loop #

            sampling_strategies = self.ss_schedule[self.cn]
            for i,b in enumerate(range(budget)):
                
                print('\n------generating sample {}------\n'.format(b+1))

                if self.vparams["random"] is not None:
                    print("Randomly sampling\n")
                    self.samples = gryffin.recommend(sampling_strategies = [1])
                elif ss is not None:
                    print('loading {} samples'.format(len(self.obs_log)))
                    print(f"\nsampling parameter(s) {', '.join(map(str, ss))}")
                    self.samples = gryffin.recommend(observations = self.obs_log, sampling_strategies = ss)
                elif isinstance(self.ss_schedule[self.cn], float):
                    print('loading {} samples'.format(len(self.obs_log)))
                    print(f"\nrun {self.cn + 1}, sampling parameter(s) {str(sampling_strategies)}")
                    self.samples = gryffin.recommend(observations = self.obs_log, sampling_strategies = [sampling_strategies])
                else:
                    print('loading {} samples'.format(len(self.obs_log)))
                    print(f"\nrun {self.cn + 1}, sampling parameter(s) {', '.join(map(str, sampling_strategies))}")
                    self.samples = gryffin.recommend(observations = self.obs_log, sampling_strategies = sampling_strategies)

                self.get_sample(model_path = model_path)
                
                print("\n\tSUGGESTIONS MADE:")
                samples_df = pd.DataFrame(self.samples)
                print(pd.DataFrame(self.samples))
                print()

            try: # to create a new campaign dictionary
                os.makedirs(f"{self.vparams['gryffin output']}/campaigns_{''.join(self.goal)}/campaign_{self.cn}/") 
            except: # pass if it already exists and overwrite
                pass

            # for benchmarking optimization settings #
            if self.suggestions == []:
                self.suggestions = self.samples
            self.suggestions_df = pd.DataFrame(self.suggestions) # create dataframe with new observations
            self.seen_df = pd.DataFrame(self.obs_log)
            self.root = f"{self.vparams['gryffin output']}/campaigns_{''.join(self.goal)}/campaign_{self.cn}"

            if self.vparams["log"] == "True":
                self.logger()
               
            self.seen_df.to_csv(f"{self.root}/seen_data.csv", index=False)
            self.suggestions_df.to_csv(f"{self.root}/suggested_observations.csv", index=False) # write dataframe to file                

        for column in self.suggestions_df.columns[:-self.num_of_objs]:
            if self.param_types[self.parameters.index(column)] == 'continuous':
                self.suggestions_df[column] = self.suggestions_df[column].astype(float).round(2)

        
    def logger(self):

            for i,e in enumerate(self.objective):
                prop = self.suggestions_df[e].tolist()
                if self.goal[i] == 'max':
                    m = max(prop)
                    try:
                        if m > max(self.progress[e]):
                            self.progress[e].append(m)
                            if self.progress["campaign"][-1] != self.cn+1:
                                self.progress["campaign"].append(self.cn+1)
                                self.progress["number of observations loaded"].append(self.seen_df.shape[0])
                    except:
                        self.progress[e].append(m)
                        self.progress["campaign"].append(self.cn+1)
                        self.progress["number of observations loaded"].append(self.seen_df.shape[0])
                else:
                    m = min(prop)
                    try:
                        if m < min(self.progress[e]):
                            self.progress[e].append(m)
                            if self.progress["campaign"][-1] != self.cn+1:
                                self.progress["campaign"].append(self.cn+1)
                                self.progress["number of observations loaded"].append(self.seen_df.shape[0])
                    except:
                        self.progress[e].append(m)
                        self.progress["campaign"].append(self.cn+1)
                        self.progress["number of observations loaded"].append(self.seen_df.shape[0])
            
            try:            
                pd.DataFrame.from_dict(self.progress).to_csv(f"{self.vparams['gryffin output']}/campaigns_{''.join(self.goal)}/opt_log.out", index=False)
            except:
                pass
            for i in self.suggestions:
                if i not in self.obs_log:
                    self.obs_log.append(i)
                    
    def gather(self, key = "archive_mo"):

        archives = []
        csv_files = []
        for directory in os.listdir():
            if key in directory:
                archive = directory
                try:
                    for directory in os.listdir(archive):
                        if "campaigns" in directory:
                            campaigns_folder = directory
                            for directory in os.listdir(archive+"/"+campaigns_folder):
                                if "campaign" in directory:
                                    csv_file = archive+"/"+campaigns_folder+"/"+directory+"/suggested_observations.csv" 
                                    csv_files.append(csv_file)
                except:
                    pass
                
        df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True) 
        df.to_csv(f"{key}_suggested_observations.csv", index=False)
        print(df)

        
    def load_descs(self):

        features = self.vparams["cat names"].split(" ")
        f_dicts = {}
        for feature in features:
            feature_file = self.desc_loc+feature+'.json'
            with open(feature_file, "r") as d:
                f_dict = json.load(d)
            if self.vparams["normalize descs"] == "True":
                f_vals = list(f_dict.values())
                f_df = pd.DataFrame(f_vals, columns=descs)
                norm_f_df=(f_df-f_df.min())/(f_df.max()-f_df.min())
                for i, key in enumerate(f_dict.keys()):
                    f_dict[key] = norm_f_df.loc[i, :].values.flatten().tolist()
            f_dicts[feature] = f_dict

        return f_dicts

class Simulator(Songbird):

    def __init__(self, data_path="", num_of_objs=None, config_file=None,
                 parameters=[], objective=[], goal=[], vparams={}, batched=False,
                 n_cpus=1, auto_desc=False, desc_loc=None, desc_key=None, output="archive_simulator",
                 architecture=None, transforms=["identity", "identity"], obs_log=None):

        super().__init__(data_path=data_path, num_of_objs=num_of_objs, config_file=config_file,
                         parameters=parameters, objective=objective, goal=goal, vparams=vparams, output=output,
                         batched=batched, n_cpus=n_cpus, auto_desc=auto_desc, desc_loc=desc_loc, desc_key=desc_key)
        
        self.olympus_out = output
        self.architecture = architecture
        self.obs_log = obs_log
        self.data = self.obs_log

        self.feature_transform = transforms[0]
        self.target_transform = transforms[1]

        self.emulator = Emulator()

    def olympus_config(self):

        folder = self.olympus_out
        config = {}
        config["constraints"] = {"parameters": "none", "measurements": "none"}
        config["parameters"] = self.parameter_list
        for i in config["parameters"]:
            if i["type"] == 'categorical':
                details = pickle.load(open(i["category_details"], 'rb'))
                i["options"] = list(details.keys())
                values = list(details.values())
                for index, e in enumerate(values):
                    values[index] = [round(val, 4) for val in e]
                i["descriptors"] = values
        config["measurements"] = self.objective_list
        for i in config["measurements"]:
            i["type"] = 'continuous'
        config["default_goal"] = "minimize"

        self.dataset_config = config

        self.dataset = Dataset(data=self.data, target_ids=self.objective)
        try:
            shutil.rmtree(f"{folder}/")
        except:
            pass
        os.mkdir(folder)
        self.dataset.data.to_csv(f"{folder}/data.csv", header=False, index=False)

        self.description = []
        if self.dataset.kind is None:
            self.description.append("Custom Dataset\n")
        else:
            self.description.append(f"{self.dataset.kind}\n")
        self.description.append("=========================================")
        self.description.append("                Summary")
        self.description.append("-----------------------------------------")
        self.description.append(f"    Number of Samples       {self.dataset.size:>10}")
        self.description.append(
            f"    Dimensionality          {len(self.parameter_list):>10}"
        )
        self.description.append(f"    Features:")
        for param in self.dataset_config["parameters"]:
            self.description.append(
                f"        {param['name']:<10}          {param['type']:>10}"
            )
        self.description.append(f"    Targets:")
        for target in self.dataset_config["measurements"]:
            self.description.append(f"        {target['name']:<10}          continuous")

        self.description = "\n".join(self.description)

        with open(f"{folder}/description.txt", "w") as f:
            f.write(self.description)
            f.write("\n")
        with open(f"{folder}/config.json", "w") as f:
            f.write(json.dumps(self.dataset_config, indent=4, sort_keys=True))

        self.dataset.param_space = ParameterSpace()
        for param in config["parameters"]:
            if param["type"] == "categorical":
                self.dataset.param_space.add(
                    ParameterCategorical(
                        name=param["name"],
                        options=param["options"],
                        descriptors=param["descriptors"],
                    )
                )
            else:
                self.dataset.param_space.add([Parameter().from_dict(param)])

            
    def architect(self):

        architecture = self.architecture
        
        if architecture is None:
            print("no architecture provided, using default")
            BNN = BayesNeuralNet()
        else:
            BNN = BayesNeuralNet()
            try:
                BNN.batch_size = architecture['batch_size']
            except:
                pass
            try:
                BNN.es_patience = architecture['es_patience']
            except:
                pass
            try:
                BNN.hidden_act = architecture['hidden_act']
            except:
                pass
            try:
                BNN.hidden_depth = architecture['hidden_depth']
            except:
                pass
            try:
                BNN.hidden_nodes = architecture['hidden_nodes']
            except:
                pass
            try:
                BNN.learning_rate = architecture['learning_rate']
            except:
                pass
            try:
                BNN.max_epochs = architecture['max_epochs']
            except:
                pass
            try:
                BNN.out_act = architecture['out_act']
            except:
                pass
            try:
                BNN.pred_int = architecture['pred_int']
            except:
                pass
            try:
                BNN.reg = architecture['reg']
            except:
                pass

        self.BNN = BNN

    def simulate(self, save=False, rmsd_threshold = 0.1, final_architecture = None, experimental_data = None):

        self.olympus_config()
        self.architect()
        print(self.BNN)
        self.emulator = Emulator(dataset = self.dataset,
                                 model = self.BNN,
                                 feature_transform = self.feature_transform,
                                 target_transform = self.target_transform
                                 )
        print("\nRunning Cross Validation\n")
        scores, FT, TT = self.emulator.cross_validate(rerun=True)
        if np.mean(scores["validate_rmsd"]) < rmsd_threshold:
            print(f"\nValidation rmsd under threshold ({np.mean(scores['validate_rmsd'])} < {rmsd_threshold}), training emulator\n")
            print(f"Final Data used\n{self.dataset.data}\n")
            lst = ['emulator', '_']
            for item in self.architecture.items():
                    lst.append(str(item[0]))
                    lst.append('_')
                    lst.append(str(item[1]))
                    lst.append('_')
            string = ''.join(lst[:-1])
            self.architecture = final_architecture
            self.architect()
            self.model_scores = self.emulator.train()
            self.emulator.save(f"{string}/emulator")
            score = self.validate(FT=FT, TT=TT)
            if score < 0.9:
                self.emulator.is_trained = False
            else:
                return f"{string}/emulator"
        else:
            print("\nRSMD threshold not met...\n")

    def validate(self, FT=None, TT=None):

        # all data observations
        data = pd.read_csv(self.data_path)
        params = data.iloc[:,:-1].values
        obj = data.iloc[:,-1].values.reshape(-1,1)

        # run emulator on all data observations
        pred_obj = self.emulator.run(params, num_samples=10)
        
        return r2_score(obj, pred_obs)
