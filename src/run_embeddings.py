#%%
# %load_ext autoreload
# %autoreload 2

# # cosinesim-1000_simmel-4-6_dimensions-32_walklength-3_numwalks-200_windowsize-15: only 5 nodes in main component
# cosinesim-2000_threshold-0%90: name missing


# from analysis.s2vcreator import S2vCreator
from analysis.embedding_eval import ParamEval, EmbMxCombinations, S2vMxvizEval
from analysis.experiments import Experiment
from analysis.nkselect import Selector

import argparse


# adjust nr nr rows in gridimages for param eval for new params


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument parser for language selection.')
    parser.add_argument('language', type=str, help='Specify the language to use.')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    language = args.language


    # Combine names of interesting networks into file
    # s = Selector(language)
    # s.get_interesting_networks()


    # Activate s2v conda environment!
    # Create embeddings for example networks with parameter grid
    # sc = S2vCreator(language=language, mode='params')
    # sc.run_combinations()

    # sc = S2vCreator(language=language, mode='params')
    # paths = sc.get_all_embedding_paths()
    # with open(f'all-embeddings-paths_{language}.csv', 'w') as f:
    #     for i in paths:
    #         f.write(f'{i}\n')

    # # Visualize different parameter combinations
    pe = ParamEval(language)
    pe.check_embeddings() ##########################
    pe.create_single_images()
    pe.create_grid_images()


    # Activate s2v conda environment!
    # # Create embeddings with selected parameters for all interesting networks
    # sc = S2vCreator(language=language, mode='run')
    # sc.run_combinations()


    # # Run matrix clustering on embeddigns
    # emc = EmbMxCombinations(language, output_dir='s2v', add_color=False, by_author=False)
    # print(emc.combinations_path)
    # # # emc.evaluate_all_combinations()
    # emc.check_data(n_features=4)


    # Create MDS visualizations of embeddings
    # ex = Experiment(language=language, cmode='mx', by_author=False, output_dir='analysis_s2v')
    # ex.run_experiments()


    # # Collect MDS visualizations for mxs with different parameters
    # me = S2vMxvizEval(language)
    # me.create_param_attr_grid()
# %%
# from analysis.embedding_eval import ParamEval, EmbMxCombinations
# language = 'eng'
# ex = Experiment(language=language, cmode='mx', by_author=False, output_dir='analysis_s2v')
# ex.run_experiments(select_exp = 'singleimage')


