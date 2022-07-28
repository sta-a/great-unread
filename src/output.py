# %%

    # def _concat_and_save_predicted_labels(self,all_validation_book_label_mapping):
    #     all_validation_book_label_mapping = pd.concat(all_validation_book_label_mapping)
    #     all_validation_book_label_mapping.to_csv(os.path.join(self.results_dir, 'examples_', self.info_string, '.csv'), index=False)
    #     return all_validation_book_label_mapping

# For an imbalanced binary classification dataset, the negative class refers to the majority class (class 0) and the positive class refers to the minority class (class 1).

# Create plots from examples file
import numpy as np
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os

out_dir = '/home/annina/scripts/great_unread_nlp/data/results/plots_ccls/'

## Best results
# filename, (stepsize, correlation, significance)
results_eng = [('eng_regression_xgboost_param-None_label-sentiart_feat-book_dimred-None_drop-3',
                  0.03, 0.233, '**'),
                ('eng_regression_xgboost_param-None_label-textblob_feat-chunk_dimred-None_drop-2',
                 0.03, -0.01, '*'),]
                #('eng_regression_xgboost_param-None_label-combined_feat-baac_dimred-None_drop-4',
                 #0.05, 0.131, '')]
results_ger = [('ger_regression_xgboost_param-None_label-sentiart_feat-book_dimred-None_drop-3', 
                  0.05, 0.198, '**'),
                ('ger_regression_xgboost_param-None_label-textblob_feat-baac_dimred-None_drop-2', 
                 0.07, 0.049, '**'),]
                #('ger_regression_xgboost_param-None_label-combined_feat-baac_dimred-None_drop-3', 
                 #0.05, 0.074, '')]

# %%
min_y = 0
max_y = 0
min_ypred = 0
max_ypred = 0
for language in ['eng', 'ger']:
    results_dir = os.path.join('/home/annina/scripts/great_unread_nlp/data_archive/results_jcls_conference', language)
    if language == 'eng':
        results = results_eng
    else:
        results = results_ger
        
    for result in results:
        filename = result[0]
        df = pd.read_csv(os.path.join(results_dir, 'examples-', filename, '.csv'))
        curr_min_y = df['y'].min()
        curr_max_y = df['y'].max()
        curr_min_ypred = df['ypred'].min()
        curr_max_ypred = df['ypred'].max()
        min_y = min(min_y, curr_min_y)
        max_y = max(max_y, curr_max_y)
        min_ypred = min(min_ypred, curr_min_ypred)
        max_ypred = max(max_ypred, curr_max_ypred)
        print(min_y,max_y,min_ypred,max_ypred)

# y on x-axis, ypred on y-axis
x_axis_limit = max(abs(min_y), max_y)
y_axis_limit = max(abs(min_ypred), max_ypred)
x_axis_limit = max(x_axis_limit, y_axis_limit)
y_axis_limit = max(x_axis_limit, y_axis_limit)

# %%
print(min_y,max_y,min_ypred,max_ypred, x_axis_limit, y_axis_limit)

# %%
for language in ['eng', 'ger']:
    results_dir = os.path.join('/home/annina/scripts/great_unread_nlp/data_archive/results_jcls_conference', language)
    if language == 'eng':
        results = results_eng
        color = 'm'
    else:
        results = results_ger
        color = 'teal'
        
    for result in results:
        filename = result[0]
        stepsize = result[1]
        correlation = result[2]
        significance = result[3]
        df = pd.read_csv(os.path.join(results_dir, 'examples-', filename, '.csv'))

        fig = plt.figure(figsize = (4,4)) 
        ax = fig.add_subplot(111)
        ax.scatter(x=df['y'], y=df['ypred'], s=9, c=color)
        #ax.grid()
        
        ax.set_xbound(lower=-x_axis_limit,upper= x_axis_limit + 0.01)
        ax.set_ybound(lower=-y_axis_limit,upper= y_axis_limit + 0.01)
        plt.draw()
        ax.tick_params(axis='x', which='both', labelsize=12, labelrotation=45)
        ax.tick_params(axis='y', which='both', labelsize=12)
        ax.xaxis.set_ticks(np.arange(-0.15, 0.2, 0.05))

        ax.text(x=-x_axis_limit + 0.02, 
                y=y_axis_limit - 0.02,
                s=f'r = {correlation}{significance}', 
                fontsize=18)
        ax.set_xlabel('Sentiment Scores', fontsize=15)
        ax.set_ylabel('Predicted Scores', fontsize=15)
        fig.savefig(f'{out_dir}{filename}.png', dpi=400, bbox_inches='tight')
        fig.show()


