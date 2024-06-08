

# %%
import sys
sys.path.append('..')
import os
import zipfile
import subprocess
import time
import shutil
from .embedding_utils import EmbeddingBase
'''
random_walks.txt and log file: paths are not set properly, find where they are stored and delete manually!!!
make sure that pickle dirs are not shared if struc2vec is run several times at the same time!
struc2vec cannot be run several times at the same time because of  random_walks.txt and log file
'''

class S2vCreator(EmbeddingBase):
    def __init__(self, language, mode=None):
        super().__init__(language, output_dir='s2v', edgelist_dir='sparsification_edgelists_s2v', mode=mode)
        self.file_string = 's2v'


    def get_params_mode_params(self):
        # Return many different parameter combinations for parameter selection
        params = {
            'dimensions': [32, 64, 128],
            'walk-length': [3, 5, 8, 15, 30, 60],
            'num-walks': [20, 50, 200],
            'window-size': [3, 5, 10, 15, 30],
            'until-layer': [5, 10],
            'OPT1': ['True'],
            'OPT2': ['True'],
            'OPT3': ['True']
        }
        return params

    # def get_params_mode_params(self):
    #     # Return many different parameter combinations for parameter selection
    #     params = {
    #         'dimensions': [32, 64, 128],
    #         'walk-length': [3, 5, 8, 15],
    #         'num-walks': [20, 50, 200],
    #         'window-size': [3, 5, 10, 15] ###################################
    #     }
    #     return params
    

    # def get_run_mode_params(self):
    #     # Return few parameter combinations for creating the embeddings for the actual data
    #     params = {
    #         'dimensions': [32],
    #         'walk-length': [15, 30],
    #         'num-walks': [200],
    #         'window-size': [3, 15, 30],
    #         'until-layer': [5],
    #         'OPT1': ['True'],
    #         'OPT2': ['True'],
    #         'OPT3': ['True']
    #     }
    #     return params

    def get_run_mode_params(self):
        # Return few parameter combinations for creating the embeddings for the actual data
        params = {
            'dimensions': [32],
            'walk-length': [15, 30],
            'num-walks': [200],
            'window-size': [15, 30],
            'until-layer': [5],
            'OPT1': ['True'],
            'OPT2': ['True'],
            'OPT3': ['True']
        }
        return params

    

    def get_bestparams_mode_params(self):
        # Return few parameter combinations for creating the embeddings for the actual data
        params = {
            'dimensions': [32],
            'walk-length': [30],
            'num-walks': [200],
            'window-size': [30],
            'until-layer': [5],
            'OPT1': ['True'],
            'OPT2': ['True'],
            'OPT3': ['True']
        }
        return params
    

    def get_params(self):
        if self.mode == 'params':
            params = self.get_params_mode_params()
        elif self.mode == 'run':
            params = self.get_run_mode_params()
        elif self.mode == 'bestparams':
            params = self.get_bestparams_mode_params()
        return params
    

    def delete_files(self, src_dir, s2v_dir, parent_dir):
        file_names = ['random_walks.txt', 'struc2vec.log']

        for file_name in file_names:
            src_file_path = os.path.join(src_dir, file_name)
            if os.path.exists(src_file_path):
                os.remove(src_file_path)
                print(f'Deleted {file_name} from src_dir.')

        for file_name in file_names:
            s2v_file_path = os.path.join(s2v_dir, file_name)
            if os.path.exists(s2v_file_path):
                os.remove(s2v_file_path)
                print(f'Deleted {file_name} from s2v_dir.')

        for file_name in file_names:
            p_file_path = os.path.join(parent_dir, file_name)
            if os.path.exists(p_file_path):
                os.remove(p_file_path)
                print(f'Deleted {file_name} from parent_dir.')

        for file_name in file_names:
            data_file_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
                print(f'Deleted {file_name} from self.data_dir.')


    def delete_dirs(self, src_dir, s2v_dir, s2v_zip_path, s2v_pkl):
        # Delete the unzipped dir
        shutil.rmtree(s2v_dir)
        print(f'Deleted the zip file at path {s2v_dir}')

        # Unzip the file
        with zipfile.ZipFile(s2v_zip_path, 'r') as zip_ref:
            zip_ref.extractall(src_dir)
            print(f'Extracted files to {src_dir}')


        if os.path.exists(s2v_pkl):
            try:
                shutil.rmtree(s2v_pkl)
                print(f'Directory {s2v_pkl} deleted successfully.')
            except Exception as e:
                print(f'Error: Failed to delete directory {s2v_pkl}: {e}')
        
        # Recreate the directory
        try:
            os.makedirs(s2v_pkl)
            print(f'Directory {s2v_pkl} created successfully.')
        except Exception as e:
            print(f'Error: Failed to create directory {s2v_pkl}: {e}')
    

    def create_embeddings(self, edgelist, embedding_path, kwargs):
        edgelist = os.path.join(self.edgelist_dir, edgelist)

        parent_dir = os.path.dirname(self.data_dir)
        src_dir = os.path.join(parent_dir, 'src')
        s2v_dir = os.path.join(src_dir, 'struc2vec-master_rwpath')
        s2v_pkl = os.path.join(s2v_dir, 'pickles')
        s2v_script = os.path.join(s2v_dir, 'src', 'main.py')
        s2v_zip_path = os.path.join(parent_dir, 'src', 'struc2vec-master_rwpath.zip')

        self.delete_dirs(src_dir, s2v_dir, s2v_zip_path, s2v_pkl)
        self.delete_files(src_dir, s2v_dir, parent_dir)


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
            '--workers', str(6)]  # 5-7 workers, less too slow, more does not produce output
        
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
        # subprocess.run(cmd)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f'Error: Command failed with return code {result.returncode}')
        else:
            print(f'Embeddings created successfully for {edgelist}')
            print(f'{time.time()-s}s to create embeddings for {edgelist}.')
            self.check_single_embedding_dimensions(edgelist, embedding_path)



    def get_param_combinations(self):
        param_combs =  super().get_param_combinations()

        # Remove combinations that differ only in the value of 'until-layer', except when OPT3 is set to True
        param_combs = [
            d for d in param_combs
            if d.get('OPT3', True) or d['until-layer'] == 5  # Keep dicts with OPT3: True or until-layer: 5
        ]
        return param_combs   

