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
    ex.run_experiments()