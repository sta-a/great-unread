

# %%
import sys
sys.path.append("..")
import os
import subprocess
import time
import shutil
from .embedding_utils import EmbeddingBase


class S2vCreator(EmbeddingBase):
    def __init__(self, language, mode=None):
        super().__init__(language, output_dir='s2v_test', edgelist_dir='sparsification_edgelists_s2v', mode=mode)
        self.file_string = 's2v'


    def get_params_params(self):
        # Return many different parameter combinations for parameter selection
        params = {
            'dimensions': [32, 64, 128],
            'walk-length': [3, 5, 8, 15],
            'num-walks': [20, 50, 200],
            'window-size': [3, 5, 10, 15]
        }
        return params
    

    def get_run_params(self):
        # Return few parameter combinations for creating the embeddings for the actual data
        params = {
            'dimensions': [32],
            'walk-length': [3, 15],
            'num-walks': [200],
            'window-size': [3, 15]
        }
        return params
    

    def get_params(self):
        if self.mode == 'params':
            params = self.get_params_params()
        elif self.mode == 'run':
            params = self.get_run_params()
        return params
    

    def create_embeddings(self, edgelist, embedding_path, kwargs):
        edgelist = os.path.join(self.edgelist_dir, edgelist)
        print('\n\nedgelist in create embeddings', edgelist)
        print('embedding_path', embedding_path)

        parent_dir = os.path.dirname(self.data_dir)
        s2v_dir = os.path.join(parent_dir, 'src', 'struc2vec-master')
        s2v_pkl = os.path.join(s2v_dir, 'pickles')
        s2v_script = os.path.join(s2v_dir, 'src', 'main.py')


        if os.path.exists(s2v_pkl):
            try:
                shutil.rmtree(s2v_pkl)
                print(f"Directory '{s2v_pkl}' deleted successfully.")
            except Exception as e:
                print(f"Error: Failed to delete directory '{s2v_pkl}': {e}")
        
        # Recreate the directory
        try:
            os.makedirs(s2v_pkl)
            print(f"Directory '{s2v_pkl}' created successfully.")
        except Exception as e:
            print(f"Error: Failed to create directory '{s2v_pkl}': {e}")




        if 'threshold' in edgelist:
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
        print(f'{time.time()-s}s to create embeddings for {edgelist}.')

