# %%
# %load_ext autoreload
# %autoreload 2
# Don't display plots
# %matplotlib agg

'''
Main script for creating similarity matrices, running clustering algorithms, and finding best results.
By setting the output_dir to 'analysis', the analyses for the unsparsified matrices and networks are run.
By setting the output_dir to 'analysis_s2v', the analyses for the position similarity matrices are run.
'''
import sys
sys.path.append("..")
import os
import argparse

from cluster.combinations import MxCombinations, NkCombinations
from analysis.experiments import Experiment
from analysis.nkselect import NkNetworkGrid, SparsGrid, Selector

import logging
logging.basicConfig(level=logging.DEBUG)


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

    # 6 attrs for normal, 7 for by author (author removed, canon-min and canon-max)
    n_features = 6 # ['gender', 'canon', 'year', 'canon-ascat', 'year-ascat', 'author']
    if by_author:
        n_features = 7 # ['gender', 'canon', 'year', 'canon-ascat', 'year-ascat', 'canon-min', 'canon-max']

    # Create similarity matrices and run clustering algorithms on unsparified matrices
    if mode == 'mxc':
        mxc = MxCombinations(language=language, add_color=False, by_author=by_author)
        mxc.evaluate_all_combinations()
        mxc.check_data(n_features=n_features) # 6 features: 'gender', 'author', 'canon', 'year', 'canon-ascat', 'year-ascat'

    # Create similarity matrices and run clustering algorithms on sparsified matrices (networks)
    elif mode == 'nkc':
        nkc = NkCombinations(language=language, add_color=False, by_author=by_author)
        nkc.evaluate_all_combinations()
        nkc.check_data(n_features=n_features)

    # Find best combinations for unsparsified matrices
    elif mode == 'mxexp':
        ex = Experiment(language=language, cmode='mx', by_author=by_author, output_dir='analysis')
        ex.run_experiments()

    # Find best combinations for networks
    elif mode == 'nkexp':
        # output_dir='analysis': eval scores will be taken from similarity dir
        ex = Experiment(language=language, cmode='nk', by_author=by_author, output_dir='analysis')
        exps = ex.get_experiments()
        ex.run_experiments()


    # elif mode == 'viz':
    #     ig = NkNetworkGrid(language, attr='canon', by_author=by_author)
    #     # ig = SparsGrid(language, attr='canon', by_author=False)
    #     ig.visualize()


    # Combine names of interesting networks into file
    # s = Selector(language)
    # s.get_interesting_networks()

    # NkNetworkGrid
    # s = Selector(language, by_author=True)
    # s.get_interesting_networks()


