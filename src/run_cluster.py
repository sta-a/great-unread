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
from analysis.nkselect import NkNetworkGrid, SparsGrid, Selector

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
    print(f"Is by_author: {by_author}")



    if mode == 'mxc':
        # xc = MxCombinationsSpars(language=language, add_color=False, by_author=by_author)
        mxc = MxCombinations(language=language, add_color=False, by_author=by_author)
        # mxc.evaluate_all_combinations()
        mxc.check_data()

        
    elif mode == 'nkc':
        nkc = NkCombinations(language=language, add_color=False, by_author=by_author)
        # nkc.evaluate_all_combinations()
        nkc.check_data()


    elif mode == 'mxexp':
        ex = Experiment(language=language, cmode='mx', by_author=by_author, output_dir='analysis')
        ex.run_experiments()


    elif mode == 'nkexp':
        # output_dir='analysis': eval scores will be taken from similarity dir
        ex = Experiment(language=language, cmode='nk', by_author=by_author, output_dir='analysis')
        ex.run_experiments()


    elif mode == 'viz':
        ig = NkNetworkGrid(language, attr='canon', by_author=by_author)
        # ig = SparsGrid(language, attr='canon', by_author=False)
        ig.visualize()


    # Combine names of interesting networks into file
    # s = Selector(language)
    # s.get_interesting_networks()

    # NkNetworkGrid
    # s = Selector(language, by_author=True)
    # s.get_interesting_networks()


