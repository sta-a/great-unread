

# %%
import sys
sys.path.append("..")
import pandas as pd
import os
from utils import DataHandler
from tqdm import tqdm
import subprocess
import time
from embedding_utils import EmbeddingBase


class S2vCreator(EmbeddingBase):
    def __init__(self, language):
        super().__init__(language, output_dir='s2v', edgelist_dir='sparsification_edgelists_s2v')
        self.file_string = 's2v'


    def get_params(self):
        params = {
            'dimensions': [32, 64, 128],
            'walk-length': [3, 5, 8, 15],
            'num-walks': [20, 50, 200],
            'window-size': [3, 5, 10, 15],
            'until-layer': [5, 10],
            'OPT1': ['True'],
            'OPT2': ['True'],
            'OPT3': ['True']
        }
        return params
    

    def create_embeddings(self, fn, kwargs):
        edgelist = os.path.join(self.edgelist_dir, fn)
        embedding_path = self.get_embedding_path(fn, kwargs)
        print(embedding_path)

        parent_dir = os.path.dirname(self.data_dir)
        s2v_dir = os.path.join(parent_dir, 'src', 'struc2vec-master', 'src')
        s2v_script = os.path.join(s2v_dir, 'main.py')

        if 'threshold' in fn:
            directed = '--undirected'
        else:
            directed = '--directed'

        cmd = ['python', s2v_script,
            '--input', edgelist,
            '--output', embedding_path,
            directed,
            '--weighted',
            '--iter', str(5),
            '--workers', str(7)]
        
        # Convert kwargs to params list in format [--key, value]
        params = []
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value is True:
                    params.append(f'--{key}')
            else:
                params.append(f'--{key}')
                params.append(str(value))

        cmd.extend(params)
        print(cmd)
        s = time.time()
        subprocess.run(cmd)
        print(f'{time.time()-s}s to create embeddings for {fn}.')


    def get_param_combinations(self):
        param_combs =  super().get_param_combinations()

        # Remove combinations that differ only in the value of 'until-layer', except when OPT3 is set to True
        param_combs = [
            d for d in param_combs
            if d.get('OPT3', True) or d['until-layer'] == 5  # Keep dicts with OPT3: True or until-layer: 5
        ]
        return param_combs   




# for language in ['eng', 'ger']:
#     ne = S2vCreator(language)
#     ne.run_combinations()
    # pc = ne.get_param_combinations()
    # for i in pc:
    #     print(i)
    # print(len(pc))

# %%
