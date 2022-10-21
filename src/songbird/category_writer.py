#!/usr/bin/env python
import os
import copy
import pickle
import numpy as np

class CategoryWriter(object):

    def __init__(self, num_opts, num_dims):
        self.num_opts = num_opts
        self.num_dims = num_dims

    def write_categories(self, parameter, options, home_dir, descriptors, with_descriptors = True):

        param_name = parameter
        options = sorted(options) # a list of strings

        if not os.path.isdir(f'{home_dir}/CatDetails'):
            os.mkdir(f'{home_dir}/CatDetails')
        cat_details_file = f'{home_dir}/CatDetails/cat_details_{param_name[0]}.pkl'

        opt_dict = {} # final form must be {option_1: list of descriptors, option_2: list of descriptors, ...}
        if not with_descriptors:
            for option in options:
                opt_dict[option] = []
        else:
            opt_dict = descriptors
            
        pickle.dump(opt_dict, open(cat_details_file, 'wb'))
