#%%
from analysis.s2vcreator import S2vCreator
from analysis.embedding_eval import ParamModeEval, EmbMxCombinations, RunModeEval, BestParamAnalyis, CombineParamEvalGrids, MirrorViz, MirrorMDSGrid
from analysis.experiments import Experiment
from analysis.analysis_utils import pklmxs_to_edgelist

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--language', type=str)
    parser.add_argument('--by_author', action='store_true')  # Boolean argument, if flag is used, by_author is set to True
    parser.add_argument('--evalcol', type=str, default='all')  # New argument with default value 'all'

    args = parser.parse_args()

    language = args.language
    by_author = args.by_author
    evalcol = args.evalcol

    print(f"Selected language: {language}")
    print(f"Is by_author: {by_author}")
    print(f"Evaluation column: {evalcol}") 


    # Convert similarity matrices to edgelist format
    params = [('sparsification_edgelists', False, True, ','), ('sparsification_edgelists_s2v', True, True, ' '), ('sparsification_edgelists_labels', False, False, ',')]
    for p in params:
        pklmxs_to_edgelist(p, by_author=by_author)


    # Activate s2v conda environment!
    # Create embeddings for example networks with parameter grid
    sc = S2vCreator(language=language, mode='params', by_author=by_author)
    sc.run_combinations(), 
    sc.check_embeddings()


    sc = S2vCreator(language=language, mode='run', by_author=by_author)
    paths = sc.get_all_embedding_paths()
    with open(f'all-embeddings-paths_{language}.csv', 'w') as f:
        for i in paths:
            f.write(f'{i}\n')
    for i in paths:
        if os.path.exists(i):
            os.remove(i)
            print(i)


    # Visualize different parameter combinations
    pe = ParamModeEval(language, by_author=by_author)
    # pe.create_single_images()
    pe.create_grid_images()
    # Combine grids for different dimensions into one image
    cg = CombineParamEvalGrids(language=language, by_author=by_author)
    cg.visualize_all()


    # Activate s2v conda environment!
    # Create embeddings with selected parameters for all interesting networks
    sc = S2vCreator(language=language, mode='run', by_author=by_author)
    sc.run_combinations()
    sc.check_embeddings()


    # # Run matrix clustering on embeddigns
    emc = EmbMxCombinations(language, output_dir='s2v', add_color=False, by_author=by_author, eval_only=True)
    emc.evaluate_all_combinations()
    n_features = 6 # ['gender', 'canon', 'year', 'canon-ascat', 'year-ascat', 'author']
    if by_author:
        n_features = 7 # ['gender', 'canon', 'year', 'canon-ascat', 'year-ascat', 'canon-min', 'canon-max']

    emc.check_data(n_features=n_features)

    # Create interactive MDS visualizations of embeddings
    # Use run_experiments.py instead
    ex = Experiment(language=language, cmode='mx', by_author=by_author, output_dir='analysis_s2v')
    ex.run_experiments(select_exp=evalcol, select_exp_from_substring=True)


    # Create interactive networks
    ex = Experiment(language=language, cmode='nk', by_author=by_author, output_dir='analysis_s2v')
    ex.run_experiments(select_exp='singleimage_analysis')

    # Collect MDS visualizations for mxs with different parameters
    me = RunModeEval(language)
    # Create one plot with all 6 images of the different param combinations, for each attr
    me.create_param_grid()


    # Place all MDS plots for the same matrix with different s2v params on top of each other, and different attributes next to each other.
    bpa = BestParamAnalyis(language)
    bpa.create_attr_grid()


    # Test mirror graph
    sc = S2vCreator(language=language, mode='mirror', by_author=by_author)
    sc.run_combinations()
    sc.check_embeddings()

    pe = MirrorViz(language, by_author=by_author)
    # pe.create_single_images()
    pe.check_mirror_node_similarity() # Check how similar mirror node is to original node
    mmg = MirrorMDSGrid(language=language, by_author=by_author) # Combine single images into grid
    mmg.visualize_all()
# %%
