#%%
# # cosinesim-1000_simmel-4-6_dimensions-32_walklength-3_numwalks-200_windowsize-15: only 5 nodes in main component

from analysis.s2vcreator import S2vCreator
from analysis.embedding_eval import ParamModeEval, EmbMxCombinations, RunModeEval, BestParamAnalyis, CombineParamEvalGrids
from analysis.experiments import Experiment
from analysis.nkselect import Selector

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--language', type=str)
    parser.add_argument('--by_author', action='store_true')  # Boolean argument, if flag is used, by_author is set to True

    args = parser.parse_args()

    language = args.language
    by_author = args.by_author

    print(f"Selected language: {language}")
    print(f"Is by_author: {by_author}")




    # # Activate s2v conda environment!
    # # Create embeddings for example networks with parameter grid
    # sc = S2vCreator(language=language, mode='params', by_author=by_author)
    # sc.run_combinations(), 
    # sc.check_embeddings()


    # sc = S2vCreator(language=language, mode='run', by_author=by_author)
    # paths = sc.get_all_embedding_paths()
    # with open(f'all-embeddings-paths_{language}.csv', 'w') as f:
    #     for i in paths:
    #         f.write(f'{i}\n')
    # for i in paths:
    #     if os.path.exists(i):
    #         os.remove(i)
    #         print(i)


    # # Visualize different parameter combinations
    pe = ParamModeEval(language, by_author=by_author)
    # pe.create_single_images()
    pe.create_grid_images()
    # # Combine grids for different dimensions into one image
    cg = CombineParamEvalGrids(language=language, by_author=by_author)
    cg.visualize_all()


    # # Activate s2v conda environment!
    # # Create embeddings with selected parameters for all interesting networks
    # sc = S2vCreator(language=language, mode='run', by_author=by_author)
    # # sc.run_combinations()
    # sc.check_embeddings()


    # # Run matrix clustering on embeddigns
    # emc = EmbMxCombinations(language, output_dir='s2v', add_color=False, by_author=by_author)
    # print(emc.combinations_path)
    # emc.evaluate_all_combinations()
    # emc.check_data(n_features=4)


    # # Create MDS visualizations of embeddings
    # ex = Experiment(language=language, cmode='mx', by_author=by_author, output_dir='analysis_s2v')
    # ex.run_experiments(select_exp='singleimage') # run first
    # ex.run_experiments()


    # # Collect MDS visualizations for mxs with different parameters
    # me = RunModeEval(language)
    # # Create one plot with all 6 images of the different param combinations, for each attr
    # me.create_param_grid()


    # # Place all MDS plots for the same matrix with different s2v params on top of each other, and different attributes next to each other.
    # bpa = BestParamAnalyis(language)
    # bpa.create_attr_grid()
