import sys
sys.path.append("..")
import argparse
from analysis.experiments import Experiment

import logging
logging.basicConfig(level=logging.DEBUG)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--language', type=str)
    parser.add_argument('--by_author', action='store_true')  # Boolean argument, if flag is used, by_author is set to True
    parser.add_argument('--output_dir', type=str, default='analysis')
    parser.add_argument('--cmode', type=str, default='mx')
    parser.add_argument('--evalcol', type=str, default='all')

    args = parser.parse_args()

    language = args.language
    by_author = args.by_author
    output_dir = args.output_dir
    cmode = args.cmode
    evalcol = args.evalcol

    print(f'Selected language: {language}')
    print(f'Is by_author: {by_author}')
    print(f'output_dir: {output_dir}')
    print(f'cmode: {cmode}')
    print(f'evalcol: {evalcol}') 



    ex = Experiment(language=language, by_author=by_author, output_dir=output_dir, cmode=cmode)
    ex.run_experiments(select_exp='singleimage_analysis', select_exp_from_substring=False)
    # topgender_vmeasure_nclust-2-5_threshold-0%8'

    # /media/annina/elements/back-to-computer-240615/data/analysis_s2v/eng/mx_topauthor_ARI_nclust-50-50_simmel-5-10
    # /media/annina/elements/back-to-computer-240615/data/analysis/eng/topyear-ascat_mean_purity_nclust-151-200
    # 
    # topauthor_ARI_nclust-50-100