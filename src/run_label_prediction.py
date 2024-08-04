# %%
import argparse
from analysis.labels_pipeline import LabelPredictCont, LabelPredict
from analysis.eval_spars_mxs import SparsmxEval


# for language in ['eng']:
#     for by_author in [False]:
#         for attr in ['author']:
#             se = SparsmxEval(language=language, by_author=by_author, attr=attr)
#             se.iterate_mxs()




# %%
def main(language, attr, by_author, test):
    print('#### Language:', language, 'attr:', attr, 'by_author:', by_author, 'test', test)
    cont_attrs = {'year', 'canon'}
    cat_attrs = {'author', 'gender'}

    # Check for conflicting parameters
    if by_author and attr == 'author':
        raise ValueError('The "by_author" parameter cannot be True when "attr" is "author".')

    if attr in cont_attrs:
        lp = LabelPredictCont(language=language, by_author=by_author, attr=attr, test=test)
        lp.regressor_pipeline()
    elif attr in cat_attrs:
        lp = LabelPredict(language=language, by_author=by_author, attr=attr, test=test)
        # lp.data_exploration()
        lp.get_upsampled_splits()
        # lp.classifier_pipeline()
    else:
        raise ValueError('Attribute should be one of "year", "canon", "author", "gender".')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run regressor or classifier pipeline with specified language and attribute.')
    parser.add_argument('--language', type=str, required=True, help='Language for the pipeline (e.g., "eng" or "ger").')
    parser.add_argument('--attr', type=str, required=True, help='Attribute for the pipeline (e.g., "year", "canon", "author", "gender").')
    parser.add_argument('--by_author', action='store_true', default=False, help='Boolean flag indicating if the classification is by author.')
    parser.add_argument('--test', action='store_true', default=False, help='Boolean flag to run with a small test sample.')

    args = parser.parse_args()

    main(args.language, args.attr, args.by_author, args.test)