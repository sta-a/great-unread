# %%
# Create plots from examples file
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

filenames_eng = {'eng_regression_xgboost_param-None_label-sentiart_feat-book_dimred-None_drop-3':0.03,
                'eng_regression_xgboost_param-None_label-textblob_feat-chunk_dimred-None_drop-2':0.03,
                'eng_regression_xgboost_param-None_label-combined_feat-baac_dimred-None_drop-4':0.05}
filenames_ger = {'ger_regression_xgboost_param-None_label-sentiart_feat-book_dimred-None_drop-3':0.05,
                'ger_regression_xgboost_param-None_label-textblob_feat-baac_dimred-None_drop-2':0.07,
                'ger_regression_xgboost_param-None_label-combined_feat-baac_dimred-None_drop-3':0.05}

for language in ['eng', 'ger']:
    if language == 'eng':
        filenames = filenames_eng
    else:
        filenames = filenames_ger
        
    for filename, stepsize in filenames.items():
        results_dir = f'../data/results_sentiment/{language}/'
        df = pd.read_csv(results_dir + 'examples-' + filename + '.csv')
if language == 'eng':
            color = 'm'
        else:
            color = 'teal'
        plt.figure(figsize=(4,4))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        #plt.xlim([min(all_labels) - 0.01, max(all_labels) + 0.01])
        #plt.ylim([min(all_predictions) - 0.01, max(all_predictions) + 0.01])
        plt.xticks(np.arange(round(min(df['y']),2) - 0.01, round(max(df['y']),2) + 0.01, stepsize))
        plt.scatter(x=df['y'], y=df['yhat'], s=6, c=color)
        plt.xlabel('True Scores', fontsize=15)
        plt.ylabel('Predicted Scores', fontsize=15)
        plt.savefig(f'{results_dir}{filename}.png', dpi=400, bbox_inches='tight')
        plt.show();


