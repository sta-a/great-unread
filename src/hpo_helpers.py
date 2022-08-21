import pandas as pd
import pickle
import statistics
import os
import numpy as np
from copy import deepcopy
from decimal import Decimal
import random
random.seed(3)
from itertools import product
from math import sqrt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


def run_gridsearch(gridsearch_dir, language, task, label_type, features, fold, columns_list, task_params, X_train, y_train):
    print(f'Inner CV: X {X_train.shape}, y {y_train.shape}')
    # Get data, set 'file_name' column as index
    cv = CustomGroupKFold(n_splits=5, stratified=task_params['stratified']).split(X_train, y_train.values.ravel())


    # for outer_fold, (train_idx, test_idx) in enumerate(cv):
        # X_train_outer, X_test_outer = X_train.iloc[train_idx], X_train.iloc[test_idx]
        # y_train_outer, y_test_outer = y_train.iloc[train_idx], y_train.iloc[test_idx]
        # print(f'\n Inner CV: X_train_outer{X_train_outer.shape}, X_test_outer{X_test_outer.shape},  y_train_outer{y_train_outer.shape}, y_test_outer{y_test_outer.shape}')
        # dfs = {'X_train_outer': X_train_outer, 'X_test_outer': X_test_outer, 'y_train_outer': y_train_outer, 'y_test_outer': y_test_outer}
        # for name, df in dfs.items():
        #     df1 = df.reset_index()[['file_name']]
        #     print(df1['file_name'].nunique())

    
    ## Parameter Grid
    # Params that are constant between grids
    constant_param_grid = {'drop_columns__columns_to_drop': columns_list}
    param_grid = deepcopy(task_params['param_grid'])  
    [d.update(constant_param_grid) for d in param_grid]

    ## Pipeline
    pipe = Pipeline(steps=[
        ('drop_columns', ColumnTransformer()),
        ('scaler', StandardScaler()),
        #('dimred', SelectPercentile()),
        ('clf', SVR())
        ])

    gridsearch = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=task_params['scoring'],
        n_jobs=-1,
        refit=task_params['refit'],
        cv=cv,
        verbose=1,
        error_score='raise', #np.nan
        return_train_score=False
    )
    gridsearch.fit(X_train, y_train.values.ravel())

    cv_results = pd.DataFrame(gridsearch.cv_results_)
    cv_results.insert(0, 'features', features)
    cv_results.insert(0, 'fold', fold)

    if task == 'regression':
        cv_results['harmonic_pvalue'] = cv_results.apply(apply_harmonic_pvalue, axis=1)

    cv_results.to_csv(
        os.path.join(gridsearch_dir, f'inner-cv_{language}_{task}_{label_type}_{features}_fold-{fold}.csv'), 
        index=False, 
        na_rep='NaN')
    with open(os.path.join(gridsearch_dir, f'gridsearch-object_{language}_{task}_{label_type}_{features}_fold-{fold}.pkl'), 'wb') as f:
        pickle.dump(gridsearch, f, -1)

    return gridsearch


class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self. columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def _drop_column(column):
            for string in self.columns_to_drop:
                if string in column:
                    return True
            return False
        if self.columns_to_drop == None:
            return X
        else:
            X_new = X[[column for column in X.columns if not _drop_column(column)]]
            dropped_cols = [column for column in X.columns if _drop_column(column)]

            # columns_before_drop = set(X.columns)
            # columns_after_drop = set(X_new.columns)
            # print(f'Dropped {len(columns_before_drop - columns_after_drop)} columns.') 
            return X_new


def analyze_cv(X, cv):
    X_fulltext = ColumnTransformer(columns_to_drop='_chunk').fit_transform(X, None)
    for train_idxs, test_idxs in cv:
        groups = get_author_groups(X_fulltext)
        X_train = X_fulltext.iloc[train_idxs]
        groups_train = groups.iloc[train_idxs]
        X_test = X_fulltext.iloc[test_idxs]
        groups_test = groups.iloc[test_idxs]
        dfs = {'X_train': X_train, 'X_test': X_test} #'groups_train': groups_train, 'groups_test': groups_test
        for name, df in dfs.items():
            print(f'Shape of {name} before removing duplicate rows: {df.shape}')
            dfs[name] = df = df[~df.index.duplicated(keep="first")]
            print(f'Shape of {name} after removing duplicate rows: {df.shape}')


def refit_regression(cv_results):
    # refit using a callable
    # find highest correlation coefficiant that has a significant harmonic p-value
    df = pd.DataFrame(cv_results)
    df_unfiltered = df.copy(deep=True)

    refit_file_path = '/home/annina/scripts/great_unread_nlp/data/nested_gridsearch/eng/best-models-in-refit.csv' #############################################3
    # Add harmonic p-value
    if not 'harmonic_pvalue' in df.columns:
        df['harmonic_pvalue'] = df.apply(apply_harmonic_pvalue, axis=1)

    # Filter df
    significance_threshold = 0.1
    df = df[~df['harmonic_pvalue'].isna()]
    df = df[df['harmonic_pvalue']<significance_threshold]

    # Check if there is any model with significant correlation coefficient
    if df.shape[0] == 0:
        print(f'No model has a significant hamonic p-value. Model with the highest correlation coefficient is returned.')
        best_idx = df_unfiltered['mean_test_corr'].idxmax()
    else:
        # Check how many models have the highest correlation coefficent
        max_metric = df['mean_test_corr'].max()
        # Find index of maximum correlation
        best_models = df.loc[df['mean_test_corr'] == max_metric]
        print('best models', best_models)
        if best_models.shape[0] > 1:
            print('Number of models that have the highest correlation coefficient and significant p-value: ', best_models.shape[0])
            with open(refit_file_path, 'a') as f:
                f.write(f'\nBest models\n')
            best_models.to_csv(refit_file_path, mode='a', header=True, index=True)

        # Find index of maximum correlation
        best_idxs = df_unfiltered.index[df_unfiltered['mean_test_corr'] == max_metric].tolist()
        best_idx = int(best_idxs[0])
        print('best idxs', best_idxs, 'best index', best_idx)

    best_idx_df = df_unfiltered.iloc[best_idx].to_frame().T
    print('best index df', type(best_idx_df), best_idx_df)
    with open(refit_file_path, 'a') as f:
        f.write(f'\nBest model with best index\n')
    best_idx_df.to_csv(refit_file_path, mode='a', header=True, index=False)
    print(f'Best index:{best_idx}')

    return best_idx
    

def apply_harmonic_pvalue(row):
    # Harmonic mean p-value
    # Takes row from GridSearchCV.cv_results_ as input
    # Match columns that contain test pvalues for each split
    pvalues = row[row.index.str.contains('split._test_corr_pvalue', regex=True)]
    try:
        denominator = sum([Decimal(1)/Decimal(x) for x in pvalues])
        harmonic_pval = len(pvalues)/denominator
    except ZeroDivisionError:
        print('Could not calculate harmonic p-value because cv p-values are 0 or approximately 0.')
        harmonic_pval = np.nan
    finally:
        return harmonic_pval


def score_regression(estimator, X, y):
    '''
    Multiple evaluation metrics for regression.
    Callable for the 'scoring' parameter in GridSearchCV. 
    '''

    y_pred = estimator.predict(X)
    corr, corr_pvalue = pearsonr(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    return {'corr': corr,
            'corr_pvalue': corr_pvalue,
            'r2': r2,
            'rmse': rmse,
            'mae': mae}

def score_binary(estimator, X, y):
    '''
    Multiple evaluation metrics for binary classification.
    Callable for the 'scoring' parameter in GridSearchCV. 
    '''
    y_pred = estimator.predict(X)
    acc = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    return {'acc': acc,
            'balanced_acc': balanced_acc}


def score_multiclass(estimator, X, y):
    '''
    Multiple evaluation metrics for multiclass classification.
    Callable for the 'scoring' parameter in GridSearchCV. 
    '''
    y_pred = estimator.predict(X)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')

    return {'f1_macro': f1_macro,
            'f1_weighted': f1_weighted}

def permute_params(model, **kwargs):
    '''
    This function takes a class and named lists of parameters: param_name = [value1, value2].
    It returns an instance of the class for every combination of the parameters of the different lists.
    '''
    param_names = list(kwargs.keys())
    params_values = list(kwargs.values())
    # All combinations of the parameters of the different lists
    combinations = list(product(*params_values))
    # Combine parameter names and parameters and instantiate class
    all_models = []
    for tup in combinations:
        x = dict()
        for i in range(0, len(tup)):
            x.update(({param_names[i]: tup[i]}))
        all_models.append(model(**x))
    return all_models


def get_labels(language, label_type, canonscores_dir=None, sentiscores_dir=None, metadata_dir=None):
    if label_type == 'canon':
        canon_file = '210907_regression_predict_02_setp3_FINAL.csv'
        labels = pd.read_csv(os.path.join(canonscores_dir, canon_file), sep=';')[['file_name', 'm3']]
        labels = labels.rename(columns={'m3': 'y'})
        labels = labels.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
        
    elif label_type == 'library':
        if language == 'eng':
            library_file = 'ENG_texts_circulating-libs.csv'
        else:
            library_file = 'GER_texts_circulating-libs.csv'
        labels = pd.read_csv(os.path.join(metadata_dir, library_file), sep=';', header=0)[['file_name', 'sum_libraries']]
        labels = labels.rename(columns={'sum_libraries': 'classified'})
        labels['classified'] = labels['classified'].apply(lambda x: 1 if x!=0 else 0)

    else:
        # Combine all labels and filenames in one file
        # File with sentiment scores for both tools
        if language == 'eng':
            scores_file = 'ENG_reviews_senti_FINAL.csv'
        else:
            scores_file = 'GER_reviews_senti_FINAL.csv'
        score_labels = pd.read_csv(os.path.join(sentiscores_dir, scores_file), sep=';', header=0)

        # File with class labels
        if language == 'eng':
            class_file = 'ENG_reviews_senti_classified.csv'
        else:
            class_file = 'GER_reviews_senti_classified.csv'
        class_labels = pd.read_csv(os.path.join(sentiscores_dir, class_file), sep=';', header=0)[['textfile', 'journal', 'file_name', 'classified']]

        labels = score_labels.merge(right=class_labels, 
                                    on=['textfile', 'journal', 'file_name'], 
                                    how='outer', 
                                    suffixes=(None, '_classlabels'), 
                                    validate='one_to_one')
        labels = labels[['sentiscore_average', 'sentiment_Textblob', 'file_name', 'classified']]

        # Sentiment Scores
        if (label_type == 'textblob') or (label_type == 'sentiart'):
            def _average_scores(group):
                textblob_value = group['sentiment_Textblob'].mean()
                sentiart_value = group['sentiscore_average'].mean()
                group['sentiment_Textblob'] = textblob_value
                group['sentiscore_average'] = sentiart_value
                return group        
            # Get one label per book
            labels = labels.groupby('file_name').apply(_average_scores)

        elif label_type == 'combined':
            def _aggregate_scores(row):
                if row['classified'] == 'positive':
                    score = row['sentiment_Textblob']
                elif row['classified'] == 'negative':
                    score = row['sentiscore_average']
                elif row['classified'] == 'not_classified':
                    score = statistics.mean([row['sentiment_Textblob'], row['sentiscore_average']]) 
                return score
            labels['combined'] = labels.apply(lambda row: _aggregate_scores(row), axis=1)
            #labels['combined'].sort_values().plot(kind='bar', figsize=(10, 5))

        elif (label_type == 'multiclass') or (label_type == 'binary'):
            #Assign one class label per work
            grouped_docs = labels.groupby('file_name')
            single_review = grouped_docs.filter(lambda x: len(x)==1)
            # If work has multiple reviews, keep labels only if reviews are not opposing (positive and negative)
            multiple_reviews = grouped_docs.filter(lambda x: len(x)>1 and not('negative' in x['classified'].values and 'positive' in x['classified'].values))
            #opposed_reviews = grouped_docs.filter(lambda x: len(x)>1 and ('negative' in x['classified'].values and 'positive' in x['classified'].values))  
            def _select_label(group):
                # Keep label with higher count, keep more extreme label if counts are equal
                count = group['classified'].value_counts().reset_index().rename(columns={'index': 'classified', 'classified': 'count'})
                if count.shape[0]>1:
                    if count.iloc[0,1] == count.iloc[1,1]:
                        grouplabel = count['classified'].max()
                    else:
                        grouplabel = count.iloc[0,0]
                    group['classified'] = grouplabel
                return group
            multiple_reviews = multiple_reviews.groupby('file_name').apply(_select_label)
            labels = pd.concat([single_review, multiple_reviews])
            labels['classified'] = labels['classified'].replace(to_replace={'positive': 3, 'not_classified': 2, 'negative': 1})

            if label_type =='binary':
                # Create label reviewed/not reviewed
                labels['classified'] = 1
            
    labels = labels.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
    labels = labels.drop_duplicates(subset='file_name')
    if label_type == 'canon':
        labels = labels.rename(columns={'m3': 'y'})
    if label_type == 'textblob':
        labels = labels[['file_name', 'sentiment_Textblob']].rename(columns={'sentiment_Textblob': 'y'})
    elif label_type == 'sentiart':
        labels = labels[['file_name', 'sentiscore_average']].rename(columns={'sentiscore_average': 'y'})
    elif (label_type == 'multiclass') or (label_type == 'binary'):
        labels = labels[['file_name', 'classified']].rename(columns={'classified': 'y'})
    elif label_type == 'combined':
        labels = labels[['file_name', 'combined']].rename(columns={'combined': 'y'})
    elif label_type == 'library':
        labels = labels[['file_name', 'classified']].rename(columns={'classified': 'y'})
    return labels


def upsample_data(df):
    upsampled_data = []
    groups = df.groupby('file_name')
    max_nr_chunks = max(groups.size())
    for name, group in groups:
        chunks_per_text = group.shape[0]
        quotient, mod = divmod(max_nr_chunks, chunks_per_text)
        quotient_groups = [group] * quotient
        mod_groups = group.sample(n=mod)
        upsampled_group = pd.concat(quotient_groups + [mod_groups])
        upsampled_data.append(upsampled_group)
    upsampled_data = pd.concat(upsampled_data)
    return upsampled_data


def get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir):
    '''
    The following file names had inconsistent spelling between versions of the data, check if errors.
    'Hegeler_Wilhelm_Mutter-Bertha_1893': 'Hegelers_Wilhelm_Mutter-Bertha_1893',
    'Hoffmansthal_Hugo_Ein-Brief_1902': 'Hoffmansthal_Hugo-von_Ein-Brief_1902'
        '''
    X = pd.read_csv(os.path.join(features_dir, f'{features}_features.csv'))

    print(f'Nr chunks for {language}: {X.shape[0]}')
    print(f'Nr texts for {language}: {len(X.file_name.unique())}')
    y = get_labels(language, label_type, canonscores_dir, sentiscores_dir, metadata_dir)
    # For english regression, 1 label is duplicated
    print(f'Nr labels for {language} {task}: {y.shape}, {y.nunique()}')
    #y_before_merge = set(y['file_name'])

    if task == 'regression':
        df = X.merge(right=y, on='file_name', how='inner', validate='many_to_one')
    else:
        # Select books written after year of first review
        # Keep only those texts that that were published after the first review had appeared
        df = X.merge(right=y, on='file_name', how='left', validate='many_to_one')
        df['y'] = df['y'].fillna(value=0)
        book_year = df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
        review_year = book_year[df['y'] != 0]
        df = df.loc[book_year>=min(review_year)]
        print('Min review year', min(review_year))

    print(f'Nr texts for {language} after combining with labels: {df.file_name.nunique()}')
    print(f'Nr labels for {language} {task} after combining with features: {df.y.shape}')
    
    # y_after_merge = set(df['file_name'])
    # print('labels difference', list(y_before_merge - y_after_merge))

    # if (features == 'cacb') or (features == 'chunk'):
    #     print('##############################################')
    #     print('Nr unique texts:', df['file_name'].value_counts())
    #     df = upsample_data(df)
    #     print('Value counts after upsampling:', pd.unique(df['file_name'].value_counts()))

    X = df.drop(labels=['y'], axis=1, inplace=False).set_index('file_name', drop=True)
    # y = df[['file_name', 'y']].set_index('file_name', drop=True)
    y = df[['y', 'file_name']].set_index('file_name', drop=True)


    print('NaN in X: ', X.isnull().values.any())
    print('NaN in y: ', y.isnull().values.any())
    print('X, y shape', X.shape, y.shape)

    return X, y


class CustomXGBClassifier(XGBClassifier):
    '''
    XGBClassifier that sets sample_weights parameter in fit()
    To be used in pipeline & grid search because fit() has common interface for all estimators
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X,y):
        weights = compute_sample_weight('balanced', y=y)
        super().fit(X, y, sample_weight=weights)

        return self


def get_author_groups(X):
    '''
    Create column with author name so that all texts by one author belong to the same group
    Account for different spelling versions of author name
    '''
    alias_dict = {
    'Hoffmansthal_Hugo': ['Hoffmansthal_Hugo-von'], 
    'Schlaf_Johannes': ['Holz-Schlaf_Arno-Johannes'],
    'Arnim_Bettina': ['Arnim-Arnim_Bettina-Gisela'],
    'Stevenson_Robert-Louis': ['Stevenson-Grift_Robert-Louis-Fanny-van-de', 
                                'Stevenson-Osbourne_Robert-Louis-Lloyde']}

    authors = X.index.to_frame(name='file_name')
    authors['author'] = authors['file_name'].str.split('_').str[:2].str.join('_')

    for author, aliases in alias_dict.items():
        for alias in aliases:
            authors['author'].replace(to_replace=alias, value=author, inplace=True)
    return authors['author']


class CustomGroupKFold():
    '''
    Split book names into n folds.
    All works of an author are put into the same fold.
    If stratified==True, the folds are created so that each fold contains approximately the same number of samples from each class.
    For classification, all labels must be represented in all classes. XGBClassifier throws an error if labels do not go from 0 to num_classes-1.
    '''
    def __init__(self, n_splits, stratified=False):
        self.n_splits = n_splits
        self.stratified = stratified

    def split(self, X, y):
        author_groups = get_author_groups(X)
        indices = []
        if self.stratified:
            cv = StratifiedGroupKFold(n_splits=self.n_splits)
            splits = cv.split(X, y, groups=author_groups)
        else:
            cv = GroupKFold(n_splits=self.n_splits)
            splits = cv.split(X, y, groups=author_groups)
        for train_idxs, test_idxs in splits:
            indices.append((train_idxs, test_idxs))

        return indices

def get_task_params(task, testing):
    # All paramters
    # Separate grids for conditional parameters
    param_grid_regression = [
        {'clf': (SVR(),),
        'clf__C': [0.1, 1],
        'clf__epsilon': [0.001, 0.01],
        'scaler': [StandardScaler()]},
        # {'clf': (Lasso(),),
        # 'clf__alpha': [1,10, 100, 1000],
        # 'clf__tol': [0.0001], # default
        # 'clf__max_iter': [1000]}, # default
        {'clf': (XGBRegressor(objective='reg:squarederror', random_state=7, n_jobs=1),),
        'clf__max_depth': [2, 4, 6, 8],
        'clf__learning_rate': [0.033, 0.1, 0.3], #0.01 does not produce result, default=0.3
        'clf__colsample_bytree': [0.33, 0.60, 0.75],
        'scaler': ['passthrough']}, 
        ]

    param_grid_binary = [
        {'clf': (SVC(class_weight='balanced'),),
        'clf__C': [0.1, 1, 10, 100, 1000]},
        {'clf': (CustomXGBClassifier(objective='binary:logistic', random_state=7, use_label_encoder=False, eval_metric='logloss', n_jobs=1),),
        'clf__max_depth': [2, 4, 6, 8],
        'clf__learning_rate': [0.01, 0.033, 0.1, 0.3],
        'clf__colsample_bytree': [0.33, 0.60, 0.75]}, 
        ]

    param_grid_multiclass= [
        {'clf': (SVC(class_weight='balanced'),),
        'clf__C': [0.1, 1, 10, 100, 1000]},
        {'clf': (CustomXGBClassifier(objective='multi:softmax', random_state=7, use_label_encoder=False, eval_metric='mlogloss', n_jobs=1),),
        'clf__max_depth': [2, 4, 6, 8],
        'clf__learning_rate': [0.01, 0.033, 0.1, 0.3],
        'clf__colsample_bytree': [0.33, 0.60, 0.75]}, 
        ]

    if testing:
        print('Using testing param grid.')
        param_grid_regression = [
            {'clf': (SVR(),),
            'clf__C': [0.1],
            'clf__epsilon': [0.001, 0.01],
            'scaler': [StandardScaler()]},
            # {'clf': (Lasso(),),
            # 'clf__alpha': [1],
            # 'clf__tol': [0.0001], # default
            # 'clf__max_iter': [1000]}, # default
            {'clf': (XGBRegressor(objective='reg:squarederror', random_state=7, n_jobs=1),),
            'clf__max_depth': [20], #20
            'clf__learning_rate': [0.033],
            'clf__colsample_bytree': [0.33, 0.60, 0.75],
            'scaler': ['passthrough']}, 
            ]

        param_grid_binary = [
            {'clf': (SVC(class_weight='balanced'),),
            'clf__C': [1]},
            {'clf': (CustomXGBClassifier(objective='binary:logistic', random_state=666, use_label_encoder=False, eval_metric='logloss', n_jobs=1),),
            'clf__max_depth': [20],
            'clf__learning_rate': [0.1],
            'clf__colsample_bytree': [0.75]}, 
            ]

        param_grid_multiclass= [
            {'clf': (SVC(class_weight='balanced'),),
            'clf__C': [1]},
            {'clf': (CustomXGBClassifier(objective='multi:softmax', random_state=777, use_label_encoder=False, eval_metric='mlogloss', n_jobs=1),),
            'clf__max_depth': [20],
            'clf__learning_rate': [0.1],
            'clf__colsample_bytree': [0.75]}, 
            ]

    # # Params for feature selection that are constant between grids
    # # Permute_params() to create separate instance of transformers for every parameter combination
    # constant_param_grid = {
    #     'dimred':
    #         ['passthrough'] + 
    #         permute_params(SelectPercentile, percentile=list(range(10, 60, 10)), score_func=[f_regression, mutual_info_regression]) +  # try between 10 and 50 % of features
    #         permute_params(PCA, n_components=[None] + list(np.arange(0.81, 1.05, 0.06).tolist())) # try None and between 81 % to 99 % of the variance
    #     }

    # [d.update(constant_param_grid) for d in param_grid]

    task_params_list = {
        'regression': {
            'labels': ['sentiart', 'textblob', 'combined'],
            'scoring': score_regression,
            'refit': refit_regression,
            'stratified': False,
            'param_grid': param_grid_regression,
            'features': ['baac', 'book', 'cacb', 'chunk'],
            },
        'binary': {
            'labels': ['binary'],
            'refit': 'balanced_acc',
            'scoring': score_binary,
            'stratified': True,
            'param_grid': param_grid_binary,
            'features': ['baac', 'book'],
            },
        'multiclass':{
            'labels': ['multiclass'],
            'refit': 'f1_macro',
            'scoring': score_multiclass,
            'stratified': True,
            'param_grid': param_grid_multiclass,
            'features': ['baac', 'book'],
            }
        }

    task_params_list['library'] = deepcopy(task_params_list['binary'])
    task_params_list['library']['labels'] = ['library']

    # Overwrite for testing 
    if testing:
        task_params_list['regression']['features'] = ['baac']
        task_params_list['regression']['labels'] = ['sentiart']
        task_params_list['binary']['features'] = ['book']
        task_params_list['library']['features'] = ['book']
        task_params_list['multiclass']['features'] = ['book']

    return task_params_list[task]