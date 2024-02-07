# %%
# %load_ext autoreload
# %autoreload 2
# Don't display plots
# %matplotlib agg

import sys
sys.path.append("..")
import os
import argparse

from cluster.combinations import MxCombinations, NkCombinations
from helpers import remove_directories

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

    args = parser.parse_args()

    language = args.language
    mode = args.mode

    print(f"Selected language: {language}")
    print(f"Selected mode: {mode}")



    if mode == 'mxc':
        # remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxeval'])
        # remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/mxcomb'])
        mxc = MxCombinations(language=language, add_color=False)
        # mxc.evaluate_all_combinations()
        mxc.check_data()
        
    elif mode == 'nkc':
        # remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkeval'])
        # remove_directories(['/home/annina/scripts/great_unread_nlp/data/similarity/eng/nkcomb'])
        nkc = NkCombinations(language=language, add_color=False)
        # nkc.evaluate_all_combinations()
        nkc.check_data()

