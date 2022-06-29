from copy import deepcopy
import pandas as pd
import numpy as np
import random
random.seed(2)
from scipy.stats import pearsonr
from math import sqrt
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight ######################################################3

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Lasso

from .split import AuthorCV

class Regression():
    '''Predict sentiment scores.'''
    def __init__(self, **kwargs):

        self.df = kwargs['df']
        self.dimensionality_reduction = kwargs['dimensionality_reduction']
        self.drop_columns = kwargs['drop_columns']
        self.eval_metric = kwargs['eval_metric']
        self.features_name = kwargs['features_name']
        self.info_string = kwargs['info_string']
        self.labels = kwargs['labels']
        self.labels_name = kwargs['labels_name']
        self.language = kwargs['language']
        self.model = kwargs['model']
        self.results_dir = kwargs['results_dir']
        self.task_type = kwargs['task_type']
        self.verbose = kwargs['verbose']

        if self.task_type == 'regression':
            self.outer_cv_split = AuthorCV(df=self.df, n_folds=10, seed=1, return_indices=False).get_folds()
        else:
            self.outer_cv_split = AuthorCV(df=self.df, n_folds=5, seed=1, stratified=True, return_indices=False).get_folds()

        assert isinstance(self.drop_columns, list)
        for i in self.drop_columns:
            assert isinstance(i, str)
        assert (self.dimensionality_reduction in ['k_best_f_reg_0_10', 'k_best_mutual_info_0_10', 'ss_pca_0_95']) or (self.dimensionality_reduction is None)
        self._assert_class_specific()

        self.df = self._drop_columns()

    def _assert_class_specific(self):
        assert self.model in ['svr', 'lasso', 'xgboost']
        assert self.features_name in ['book', 'chunk', 'baac', 'cacb']
        assert self.labels_name in ['textblob', 'sentiart', 'combined']

    def _combine_df_labels(self, df):
        #Average of sentiscores per book
        df = df.merge(right=self.labels, on='file_name', how='inner', validate='many_to_one')
        return df

    def _concat_and_save_predicted_labels(self,all_validation_book_label_mapping):
        all_validation_book_label_mapping = pd.concat(all_validation_book_label_mapping)
        all_validation_book_label_mapping.to_csv(f'{self.results_dir}examples-{self.info_string}.csv', index=False)
        return all_validation_book_label_mapping

    def _custom_pca(self, train_X):
        for i in range(5, train_X.shape[1], int((train_X.shape[1] - 5) / 10)):
            pca = PCA(n_components=i)
            new_train_X = pca.fit_transform(train_X)
            if pca.explained_variance_ratio_.sum() >= 0.95:
                break
        return new_train_X, pca

    def _drop_columns(self):
        def _drop_column(column):
            for string in self.drop_columns:
                if string in column:
                    return True
            return False

        columns_before_drop = set(self.df.columns)
        if self.drop_columns:
            df = self.df[[column for column in self.df.columns if not _drop_column(column)]].reset_index(drop=True)
        columns_after_drop = set(self.df.columns)
        if self.verbose:
            print(f'Dropped {len(columns_before_drop - columns_after_drop)} columns.') 
        return df

    def _get_model(self, train_df, train_X, train_y):
        if self.task_type == 'regression':
            inner_cv_split = AuthorCV(df=train_df, n_folds=5, seed=16, stratified=False, return_indices=True).get_folds()
        else:
            inner_cv_split = AuthorCV(df=train_df, n_folds=5, seed=16, stratified=True, return_indices=True).get_folds()

        if self.model == 'svr':
            estimator = SVR()
            param_grid={'C': [1]}
        elif self.model == 'lasso': # [1, 4]
            estimator =  Lasso()
            param_grid={'alpha': [1]}
        elif self.model == 'svc':
            estimator=SVC(class_weight='balanced'),
            param_grid={'C': [0.1, 1, 10, 100, 1000]}
        elif self.model == 'xgboost':
            param_grid = {'max_depth': [2], #4, 6, 8],  ###############################################3
                        'learning_rate': [None], #0.01, 0.033, 0.1],
                        'colsample_bytree': [0.33]} #, 0.60, 0.75]}

            if self.task_type =='regression':
                params = {'objective': 'reg:squarederror', 'random_state': 42}# if classification, maximize f1/acc score. ###################
                estimator = XGBRegressor().set_params(**params)
            elif self.task_type == 'binary' or self.task_type == 'library':
                params = {'objective': 'binary:logistic', 'random_state': 42}
                estimator=XGBClassifier().set_params(**params)
            elif self.task_type == 'multiclass':
                params = {'objective': 'multi:softmax', 'num_class': 4, 'random_state': 42}
                estimator=XGBClassifier().set_params(**params)
            
        clf = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=self._score_task,
            n_jobs=-1, ################################3
            refit=self.eval_metric,
            cv=inner_cv_split,
            verbose=3,
            #pre_dispatch= #####################################,
            return_train_score=False
        )
        clf.fit(train_X, train_y)
        return clf
        
    
    def _get_pvalue(self, validation_corr_pvalues):
        # Harmonic mean p-value
        denominator = sum([1/x for x in validation_corr_pvalues])
        mean_p_value = len(validation_corr_pvalues)/denominator
        return mean_p_value

    def _impute(self, train_X, validation_X):
        imputer = KNNImputer()
        train_X = imputer.fit_transform(train_X)
        validation_X = imputer.transform(validation_X)
        return train_X, validation_X

    def _split_df(self, df, split):
        ''' Split df into training and validation X and y for cv. '''
        train_df = df[~df['file_name'].isin(split)]
        train_X = train_df.drop(columns=['y', 'file_name'], inplace=False).values
        train_y = train_df['y'].values.ravel()

        validation_df = df[df['file_name'].isin(split)]
        validation_X = validation_df.drop(columns=['y', 'file_name'], inplace=False).values
        return train_df, train_X, train_y, validation_df, validation_X


    def _prepare_dfs(self, df, split):
        # Split data into train and validation set
        train_df, train_X, train_y, validation_df, validation_X = self._split_df(df, split)

        # Impute missing values
        train_X, validation_X = self._impute(train_X, validation_X)
        #if self.verbose:
        #    print(f'train_X.shape before {self.dimensionality_reduction}: {train_X.shape}, validation_X.shape before {self.dimensionality_reduction}: {validation_X.shape}')

        # Reduce dimensions
        train_X, validation_X = self._reduce_dimensions(train_X, train_y, validation_X)
        #if self.verbose:
        #    print(f'train_X.shape after {self.dimensionality_reduction}: {train_X.shape}, validation_X.shape after {self.dimensionality_reduction}: {validation_X.shape}')

        # Keep file names plus labels
        train_book_label_mapping = deepcopy(train_df[['file_name', 'y']])
        validation_book_label_mapping = deepcopy(validation_df[['file_name', 'y']])
        #if self.verbose:
            #print('Class distribution over train and validation set :', train_df['y'].value_counts()'\n', validation_df['y'].value_counts())
        return train_df, train_X, train_y, validation_X, train_book_label_mapping, validation_book_label_mapping

    def _reduce_dimensions(self, train_X, train_y, validation_X):
        if self.dimensionality_reduction == 'ss_pca_0_95':
            ss = StandardScaler()
            train_X = ss.fit_transform(train_X)
            validation_X = ss.transform(validation_X)
            train_X, pca = self._custom_pca(train_X)
            validation_X = pca.transform(validation_X)
        elif self.dimensionality_reduction == 'k_best_f_reg_0_10':
            k_best = SelectKBest(f_regression, k=np.minimum(int(0.10 * train_X.shape[0]), train_X.shape[1]))
            train_X = k_best.fit_transform(train_X, train_y)
            validation_X = k_best.transform(validation_X)
        elif self.dimensionality_reduction == 'k_best_mutual_info_0_10':
            k_best = SelectKBest(mutual_info_regression, k=np.minimum(int(0.10 * train_X.shape[0]), train_X.shape[1]))
            train_X = k_best.fit_transform(train_X, train_y)
            validation_X = k_best.transform(validation_X)
        elif self.dimensionality_reduction is None:
            pass
        return train_X, validation_X

    def _score_task(self, y, y_hat):
        '''
        Calculate task-specific evaluation metrics.
        '''
        corr, corr_p_value= pearsonr(y,y_hat)
        r2 = r2_score(y, y_hat)
        rmse = sqrt(mean_squared_error(y, y_hat))
        mae = mean_absolute_error(y, y_hat)
        
        return {'corr': corr,
                'corr_p_value': corr_p_value,
                'r2': r2,
                'rmse': rmse,
                'mae': mae}
    
    def run(self):
        all_validation_book_label_mapping = []
        all_train_scores = []
        all_validation_scores = []

        df = self._combine_df_labels(self.df)
        #-----------------------------------------------------------------------------------------------------------------------------
        # Outer CV
        for index, split in enumerate(self.outer_cv_split):

            # Split df
            train_df, train_X, train_y, validation_X, train_book_label_mapping, validation_book_label_mapping = self._prepare_dfs(df, split) ########3 delete train_df
            
            # Train model
            clf = self._get_model(train_df, train_X, train_y)
            
            # Predict           
            train_book_label_mapping['yhat'] = clf.predict(train_X)
            validation_book_label_mapping['yhat'] = clf.predict(validation_X)
            
            # Evaluate
            train_book_label_mapping = train_book_label_mapping.groupby('file_name').mean()
            validation_book_label_mapping = validation_book_label_mapping.groupby('file_name').mean()
            all_validation_book_label_mapping.append(validation_book_label_mapping.reset_index())
                      
            train_scores = self._score_task(train_book_label_mapping['y'].tolist(), train_book_label_mapping['yhat'].tolist())
            validation_scores = self._score_task(validation_book_label_mapping['y'].tolist(), validation_book_label_mapping['yhat'].tolist())

            all_train_scores.append(train_scores)
            all_validation_scores.append(validation_scores)
                      
            if self.verbose:
                print(f'Fold: {index+1}')
                print(train_scores)
                print(validation_scores)
        #-----------------------------------------------------------------------------------------------------------------------------
        
        all_validation_book_label_mapping = self._concat_and_save_predicted_labels(all_validation_book_label_mapping)
        if self.task_type != 'regression':
            self._make_crosstabs(all_validation_book_label_mapping)
        all_train_scores = pd.DataFrame(all_train_scores).add_prefix('train_')
        all_validation_scores = pd.DataFrame(all_validation_scores).add_prefix('validation_')

        mean_train_scores = all_train_scores.mean(axis=1)
        if self.task_type == 'regression':
            mean_train_scores['mean_p_value'] = self._get_pvalue(all_train_scores['train_corr_p_value'])
        mean_train_scores.round(3)

        mean_validation_scores = all_validation_scores.mean(axis=1)
        mean_validation_scores['mean_p_value'] = self._get_pvalue(all_validation_scores['validation_corr_p_value'])
        mean_validation_scores.round(3)
        
        if self.verbose:
            print(mean_train_scores.to_string())
            print(mean_validation_scores.to_string())

        return [mean_train_scores, mean_validation_scores]


class BinaryClassification(Regression):
    ''' Classify into reviewed/not reviewed.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _assert_class_specific(self):
        assert self.model in ['svc', 'xgboost']
        assert self.features_name in ['book', 'baac']
        assert self.labels_name in ['binary', 'library']

    def _combine_df_labels(self, df):
        #Reviews zum englischen Korpus beginnnen mit 1759 und decken alles bis 1914 ab
        df['file_name'].to_csv('test.txt')
        df = df.merge(right=self.labels, on='file_name', how='left', validate='many_to_one')
        #print('df value counts', df['y'].value_counts())
        df_filenames = df[df['y']==1]['file_name']
        labels_filenames = self.labels['file_name']
        print(set(labels_filenames) - set(df_filenames))
        #[print(x) if x not in df_filenames else '' for x in labels_filenames]
        #print('labels value counts', self.labels['y'].value_counts())
        df['y'] = df['y'].fillna(value=0)
        #Select books written after year of first review)
        book_year = df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
        review_year = book_year[df['y'] != 0]
        # Keep only those texts that that were published after the first review had appeared
        df = df.loc[book_year>=min(review_year)]
        #print(df['y'].value_counts())
        return df

    def _make_crosstabs(self, all_validation_book_label_mapping):
        crosstab = pd.crosstab(all_validation_book_label_mapping['y'], all_validation_book_label_mapping['yhat'], rownames=['True'], colnames=['Predicted'], margins=True)
        crosstab.to_csv(f'{self.results_dir}crosstab_{self.info_string}.csv', index=True)
        print('--------------------------\nCrosstab\n', crosstab, '\n--------------------------')
    
    def _score_task(y, y_hat):
        acc = accuracy_score(y, y_hat)
        balanced_acc = balanced_accuracy_score(y, y_hat)
        return {'acc': acc,
                'balanced_acc': balanced_acc}
                            

class MulticlassClassification(BinaryClassification):
    '''Classify into not reviewed/negative/not classified/positive.'''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _assert_class_specific(self):
        assert self.model in ['svc', 'xgboost']
        assert self.features_name in ['book', 'baac']
        assert self.labels_name in ['multiclass']

    def _score_task(y, y_hat):
        f1 = f1_score(y, y_hat, average='macro')
        return {'f1': f1}