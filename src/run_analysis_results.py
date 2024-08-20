from analysis.interactive_results import InteractiveResults

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


# subdirs = ['MxNkAnalysis', 'NkAnalysis']
# for language in ['eng']:
#     for subdir in ['MxNkAnalysis']:
#         ir = InteractiveResults(language=language, subdir=subdir, by_author=True)
#         ir.filter_df()



ir = InteractiveResults(language='eng', subdir='NkAnalysis', by_author=True)
ir.filter_df()