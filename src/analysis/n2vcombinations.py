# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")
import pandas as pd
import pickle
import os
import sklearn
import networkx as nx
import matplotlib as plt
import pandas as pd
from utils import DataHandler
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from n2vcreator import N2VCreator
from cluster.combinations import CombinationsBase, MxCombinations
from cluster.cluster_utils import CombinationInfo
from cluster.cluster import ClusterBase
from cluster.create import D2vDist



class N2VDist(D2vDist):
    def __init__(self, language, modes):
            super().__init__(language, output_dir='n2v', modes=modes)
            self.file_string = 'n2v' 
            self.nc = N2VCreator(self.language)

    def get_embeddings_dict(self, mode):
        emb_dict = {}
        df = self.nc.load_embeddings(mode)
        for file_name, row in df.iterrows():
            emb_dict[file_name] = np.array(row)
        return emb_dict
    
    
# class N2VCombinationsBase(CombinationsBase):
#     def __init__(self, language):
#         super().__init__(output_dir='n2v', add_color=False, cmode='mx', by_author=False)


    # def extract_info(self, file_name):
    #     # Convert file name info CombinationInfo object
    #     repdict = {'argamon_quadratic': 'argamonquadratic', 'argamon_linear': 'argamonlinear'}
    #     for key, val in repdict.items():
    #         if key in file_name:
    #             file_name = file_name.replace(key, val)

    #     mxname, sparsmode = file_name.split('_')

    #     for key, val in repdict.items():
    #         if val in mxname:
    #             mxname = mxname.replace(val, key)

    #     info = CombinationInfo(mxname=mxname, sparsmode=sparsmode)
    #     print(info.as_string())
    #     return info
    

# line 46, 208, 209, 245 in evaluate assert!! #########################

class N2VMxCombinations(MxCombinations):
    def __init__(self, language, output_dir='n2v', add_color=False, by_author=False):
        super().__init__(language, output_dir=output_dir, add_color=add_color, by_author=by_author)
    
    def load_mxs(self):
        nc = N2VCreator(self.language)
        embedding_files = [file for file in os.listdir(nc.subdir) if file.endswith('embeddings')]
        embedding_files = [x.split('.')[0] for x in embedding_files] # remove '.embeddings'
        n2vd = N2VDist(language=self.language, modes=embedding_files)
        mxs = n2vd.load_all_data(use_kwargs_for_fn='mode', file_string=n2vd.file_string, subdir=True)

        mxs_list = []
        for name, mx in mxs.items():
            mx.name = name
            print(name)
            mxs_list.append(mx)
        return mxs_list

nc = N2VMxCombinations(language='eng', add_color=False)
nc.evaluate_all_combinations()
# %%
