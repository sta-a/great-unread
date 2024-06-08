# %%
# %load_ext autoreload
# %autoreload 2
# Don't display plots
# %matplotlib agg

import sys
sys.path.append("..")
import os
import argparse

from cluster.combinations import MxCombinations, NkCombinations, MxCombinationsSpars
from helpers import remove_directories
from analysis.experiments import Experiment

import logging
logging.basicConfig(level=logging.DEBUG)





# logfiles = ['/home/annina/scripts/great_unread_nlp/data/similarity/eng/log_clst.txt']
# for i in logfiles:
#     if os.path.exists(i):
#         os.remove(i)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--language', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--by_author', action='store_true')  # Boolean argument, if flag is used, by_author is set to True

    args = parser.parse_args()

    language = args.language
    mode = args.mode
    by_author = args.by_author

    print(f"Selected language: {language}")
    print(f"Selected mode: {mode}")



    if mode == 'mxc':
    #     # remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxeval'])
    #     # remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxcomb'])
        # 
        # xc = MxCombinationsSpars(language=language, add_color=False, by_author=by_author)
        mxc = MxCombinations(language=language, add_color=False, by_author=by_author)
        mxc.evaluate_all_combinations()
        mxc.check_data()
        
    elif mode == 'nkc':
        # remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkeval'])
        # remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkcomb'])
        # nkc = NkCombinations(language=language, add_color=False, by_author=by_author)
        # nkc.evaluate_all_combinations()
        # nkc.check_data(n_features=4)


        # output_dir='analysis': eval scores will be taken from similarity dir
        ex = Experiment(language=language, cmode='nk', by_author=False, output_dir='analysis')
        ex.run_experiments()