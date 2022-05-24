from copy import deepcopy
import pandas as pd
import numpy as np
import random
random.seed(2)
from collections import Counter
from scipy.stats import pearsonr
from math import sqr
import statistics
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost
from .split import CVFolds



global_list = []
class XGBoostModel():
    '''
    This class serves as the estimator for RFE. 
    '''
    def __init__(self, train_df=None, task_type):
        self.train_df = train_df
        self.task_type = task_type
        self.is_classification = None
        self.class_weights = None

        if task_type == 'binary_classification':
            self.is_classification = True
            self.class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=train_y.astype(int).tolist())))
        elif task_type == 'multiclass_classification':
            self.is_classification = True
            self.class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1, 2, 3], y=train_y.astype(int).tolist())))
        elif task_type == 'regression':
            self.is_classification = False
        else:
            raise Exception('Not a valid task_type')

    def feval(self, preds, train_data):
        labels = train_data.get_label()
        if self.is_classification:
            labels = labels.astype(int)
            preds = preds.argmax(axis=1).astype(int)
            if task_type == 'binary_classification':
                return 'acc', accuracy_score(labels, preds)
            elif task_type == 'multiclass_classification':
                return 'f1', f1_score(labels, preds, average='macro')
        else:
            return 'corr', pearsonr(labels, preds)[0]  
    
    def fit(self, train_X, train_y):
        global global_list
        n_folds = 5
        
        print(f'Current shape of feature matrix:', train_X.shape)
        num_boost_round = 99999

                            
        train_X = deepcopy(train_X)
        train_y = deepcopy(train_y)
        if self.is_classification:
            dtrain = xgboost.DMatrix(train_X, label=train_y.astype(int), weight=[class_weights[int(i)] for i in train_y])
        else:
            dtrain = xgboost.DMatrix(train_X, label=train_y)

        all_results = []
        for max_depth in [4]: # [2, 4, 6, 8]
            for learning_rate in [0.03]: #[None, 0.01, 0.03, 0.1]
                for colsample_bytree in [0.33]: # [0.33, 0.60, 0.75]
                    for min_child_weight in [6]:
                        if task_type == 'multiclass_classification':
                            params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'n_jobs': -1, 'objective': 'multi:softmax', 'num_class': 4, 'eval_metric': 'mlogloss'}
                        elif task_type == 'binary_classification':
                            params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'n_jobs': -1, 'objective': 'multi:softmax', 'num_class': 2, 'eval_metric': 'mlogloss'}
                        elif task_type == 'regression':
                            params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'n_jobs': -1}
                        else:
                            raise Exception('Not a valid task_type')

                        cv_results = xgboost.cv(
                            params,
                            dtrain,
                            num_boost_round=num_boost_round,
                            seed=40,
                            nfold=n_folds,
                            folds=CVFolds(self.train_df, n_folds, seed=3, stratified=False, return_indices=True).split(),
                            feval=self.feval,
                            maximize=self.is_classification, # if classification, maximize f1/acc score
                            early_stopping_rounds=5, #10
                            verbose_eval=False
                        )
                        print(train_X.shape[1], max_depth, learning_rate, colsample_bytree, min_child_weight, np.round(float(cv_results['test-corr-mean'].iloc[len(cv_results['test-corr-mean'])-1]), 4))
                        all_results.append((max_depth, learning_rate, colsample_bytree, min_child_weight, cv_results))
        all_results_df = pd.DataFrame(all_results, columns=['max_depth', 'learning_rate', 'colsample_bytree', 'min_child_weight', 'cv_results'])
        all_results_df['best_validation_corr'] = all_results_df['cv_results'].apply(lambda x: x.iloc[len(x)-1]['test-corr-mean'])
        all_results_df = all_results_df.sort_values(by='best_validation_corr', ascending=False).reset_index(drop=True)
        best_parameters = all_results_df.iloc[0]
        best_max_depth = int(best_parameters['max_depth'])
        best_learning_rate = float(best_parameters['learning_rate'])
        best_colsample_bytree = float(best_parameters['colsample_bytree'])
        best_min_child_weight = float(best_parameters['min_child_weight'])
        best_validation_corr = float(best_parameters['best_validation_corr'])
        best_num_boost_round = int(len(best_parameters['cv_results']))
        
        print('Current best max_depth:', best_max_depth)
        print('Current best learning_rate:', best_learning_rate)
        print('Current best colsample_bytree:', best_colsample_bytree)
        print('Current best min_child_weight:', best_min_child_weight)
        print('Current best num_boost_round:', best_num_boost_round)
        print(f'Current best validation corr score is {np.round(best_validation_corr, 4)}')
        print('############################')
        params = {'max_depth': best_max_depth, 'learning_rate': best_learning_rate, 'colsample_bytree': best_colsample_bytree, 'min_child_weight': best_min_child_weight, 'n_jobs': -1}
        
        train_X = deepcopy(train_X)
        train_y = deepcopy(train_y)
        dtrain = xgboost.DMatrix(train_X, label=train_y)
        model = xgboost.train(params,
                              dtrain,
                              num_boost_round=best_num_boost_round,
                              verbose_eval=False)
        feature_importances = model.get_score(importance_type='gain')
        feature_importances_list = []
        for i in range(train_X.shape[1]):
            current_key = f'f{i}'
            if current_key in feature_importances:
                feature_importances_list.append(feature_importances[current_key])
            else:
                feature_importances_list.append(0.0)
        self.model = model
        self.feature_importances_ = np.array(feature_importances_list)
        
        global_list.append((best_validation_corr, train_X.shape[1], best_num_boost_round, params))
        
        return self
    
    def predict(self, X):
        return self.model.predict(xgboost.DMatrix(X))

    def predict_proba(self, X):
        return self.predict(X)
    
    def decision_function(self, X):
        return self.predict(X)
    
    def transform(self, X):
        return self.predict(X)

    def score(self, X=None, y=None):
        return 0.0

    def get_params(self, deep=True):
        return {'train_df': self.train_df}

    def set_params(self, **params):
        self.train_df = params['train_df']
        return self

    def _get_tags(self):
        return {'allow_nan': True}


class Regression():
    '''Predict sentiment scores.'''
    def __init__(self, results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True):

        self.results_dir = results_dir
        self.language = language
        self.task_type = task_type
        self.model = model
        self.model_param = model_param
        self.labels_string = labels_string
        self.labels = labels
        self.features_string = features_string
        self.df = df
        self.dimensionality_reduction = dimensionality_reduction
        self.drop_columns = drop_columns
        self.verbose = True
        self.model_info_string = f'{self.language}_{self.task_type}_{self.model}_param-{self.model_param}_label-{self.labels_string}_feat-{self.features_string}_dimred-{self.dimensionality_reduction}_drop-{len(self.drop_columns)}'

        assert isinstance(self.drop_columns, list)
        for i in self.drop_columns:
            assert isinstance(i, str)
        assert (self.dimensionality_reduction in ['k_best_f_reg_0_10', 'k_best_mutual_info_0_10', 'ss_pca_0_95', 'rfe']) or (self.dimensionality_reduction is None)
        self._check_class_specific_assertions()

        self.df = self._drop_columns()

    def _check_class_specific_assertions(self):
        assert self.model in ['xgboost', 'svr', 'lasso']
        assert self.features_string in ['book', 'chunk', 'baac', 'cacb']
        assert self.labels_string in ['canon', 'textblob', 'sentiart', 'combined']
    
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
    
    def _custom_pca(self, train_X):
        for i in range(5, train_X.shape[1], int((train_X.shape[1] - 5) / 10)):
            pca = PCA(n_components=i)
            new_train_X = pca.fit_transform(train_X)
            if pca.explained_variance_ratio_.sum() >= 0.95:
                break
        return new_train_X, pca

    def _impute(self, train_X, validation_X):
        imputer = KNNImputer()
        train_X = imputer.fit_transform(train_X)
        validation_X = imputer.transform(validation_X)
        return train_X, validation_X

    def _reduce_dimensions(self, train_X, train_y, validation_X, train_df):
        n_features_to_select = np.minimum(int(0.10 * train_X.shape[0]), train_X.shape[1])
        if self.dimensionality_reduction is None:
            pass
        elif self.dimensionality_reduction == 'ss_pca_0_95':
            ss = StandardScaler()
            train_X = ss.fit_transform(train_X)
            validation_X = ss.transform(validation_X)
            train_X, pca = self._custom_pca(train_X)
            validation_X = pca.transform(validation_X)
        elif self.dimensionality_reduction == 'k_best_f_reg_0_10':
            k_best = SelectKBest(f_regression, k=n_features_to_select)
            train_X = k_best.fit_transform(train_X, train_y)
            validation_X = k_best.transform(validation_X)
        elif self.dimensionality_reduction == 'k_best_mutual_info_0_10':
            k_best = SelectKBest(mutual_info_regression, k=n_features_to_select)
            train_X = k_best.fit_transform(train_X, train_y)
            validation_X = k_best.transform(validation_X)
        elif self.dimensionality_reduction == 'rfe':
            global global_list
            best_num_boost_round = None
            hpo = HyperParameterOptimizer(train_df)
            rfe = RFE(hpo, step=50, n_features_to_select=n_features_to_select, verbose=0) # only 14 features if all columns are dropped
            rfe.fit(train_X, train_y)
            global_list = sorted(global_list, key=lambda x: -x[0])
            best_feature_count = global_list[0][1]
            best_num_boost_round = global_list[0][2]
            best_params = global_list[0][3]
            for ranking in range(rfe.ranking_.max()):
                if (rfe.ranking_ <= ranking).sum() == best_feature_count:
                    break
            train_X = train_X[:, np.argwhere((rfe.ranking_ <= ranking) == True).T[0]]
            validation_X = validation_X[:, np.argwhere((rfe.ranking_ <= ranking) == True).T[0]]
            global_list = []
        return train_X, validation_X
    
    def _get_model(self, model_param=None, train_X=None, train_y=None, train_df=None, task_type=None):
        # Classification
        if self.model == 'svc':
            return SVC(C=model_param, class_weight='balanced')
        # Regression
        elif self.model == 'svr':
            return SVR(C=model_param)
        elif self.model == 'lasso':
            return Lasso(alpha=model_param)

        elif self.model == 'xgboost':


                        if task_type == 'binary_classification':
                            nested_cv_score = cv_results.iloc[len(cv_results)-1]['test-acc-mean']
                        elif task_type == 'multiclass_classification':
                            nested_cv_score = cv_results.iloc[len(cv_results)-1]['test-f1-mean']
                        elif task_type == 'regression':
                            nested_cv_score = cv_results.iloc[len(cv_results)-1]['test-rmse-mean']
                        else:
                            raise Exception('Not a valid task_type')
                        num_boost_round = len(cv_results)
                        if task_type == 'multiclass_classification':
                            results.append({'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'num_boost_round': num_boost_round, 'nested_cv_score': nested_cv_score, 'objective': 'multi:softmax', 'num_class': 4, 'eval_metric': 'mlogloss'})
                        elif task_type == 'binary_classification':
                            results.append({'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'num_boost_round': num_boost_round, 'nested_cv_score': nested_cv_score, 'objective': 'multi:softmax', 'num_class': 2, 'eval_metric': 'mlogloss'})
                        elif task_type == 'regression':
                            results.append({'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'num_boost_round': num_boost_round, 'nested_cv_score': nested_cv_score})
                        else:
                            raise Exception('Not a valid task_type')
            best_parameters = sorted(results, key=lambda x: x['nested_cv_score'], reverse=self.is_classification)[0]
            return best_parameters
        
    
    def _get_pvalue(self, validation_corr_pvalues):
        # Harmonic mean p-value
        denominator = sum([1/x for x in validation_corr_pvalues])
        mean_p_value = len(validation_corr_pvalues)/denominator
        return mean_p_value
    
    def _combine_df_labels(self, df):
        #Average of sentiscores per book
        df = df.merge(right=self.labels, on='file_name', how='inner', validate='many_to_one')
        return df

    def _prepare_dfs(self, split, df):
        # Prapare data
        train_df = df[~df['file_name'].isin(split)]
        train_X = train_df.drop(columns=['y', 'file_name'], inplace=False).values
        train_y = train_df['y'].values.ravel()

        validation_df = df[df['file_name'].isin(split)]
        validation_X = validation_df.drop(columns=['y', 'file_name'], inplace=False).values

        train_X, validation_X = self._impute(train_X, validation_X)
        #if self.verbose:
        #    print(f'train_X.shape before {self.dimensionality_reduction}: {train_X.shape}, validation_X.shape before {self.dimensionality_reduction}: {validation_X.shape}')
        train_X, validation_X = self._reduce_dimensions(train_X, train_y, validation_X, train_df)
        #if self.verbose:
        #    print(f'train_X.shape after {self.dimensionality_reduction}: {train_X.shape}, validation_X.shape after {self.dimensionality_reduction}: {validation_X.shape}')

        train_labels = deepcopy(train_df[['file_name', 'y']])
        validation_labels = deepcopy(validation_df[['file_name', 'y']])
        #if self.verbose:
            #print('Class distribution over train and validation set :', train_df['y'].value_counts()'\n', validation_df['y'].value_counts())
        return train_df, train_X, train_y, validation_X, train_labels, validation_labels
    
    def _concat_and_save_examples(self,all_validation_labels):
        all_validation_labels = pd.concat(all_validation_labels)
        all_validation_labels.to_csv(f'{self.results_dir}examples-{self.model_info_string}.csv', index=False)
        return all_validation_labels

    def run(self):
        # List of all predicted values/all true labels, for plotting
        all_predictions = []
        all_labels = []
        # List of dfs with file_name, y, yhat
        all_validation_labels = []

        train_mses = []
        train_maes = []
        train_r2s = []
        train_corrs = []
        
        validation_mses = []
        validation_maes = []
        validation_r2s = []
        validation_corrs = []
        validation_corr_pvalues = []

        df = self.df
        df = self._combine_df_labels(df)
        file_names_split = CVFolds(df=df, n_folds=10, seed=1, return_indices=False).split()
        
        for index, split in enumerate(file_names_split):
            print(split)
            train_df, train_X, train_y, validation_X, train_labels, validation_labels = self._prepare_dfs(split, df)

            # Train model and predict
            if self.model == 'xgboost':
                best_parameters = self._get_model(self.model_param, train_X, train_y, train_df, task_type='regression')
                dtrain = xgboost.DMatrix(train_X, label=train_y)
                num_boost_round = best_parameters['num_boost_round']
                best_parameters.pop('nested_cv_score')
                best_parameters.pop('num_boost_round')
                model = xgboost.train(best_parameters,
                                      dtrain,
                                      num_boost_round=num_boost_round,
                                      verbose_eval=False)
                train_labels['yhat'] = model.predict(xgboost.DMatrix(train_X))
                validation_labels['yhat'] = model.predict(xgboost.DMatrix(validation_X))
            else:
                model = self._get_model(self.model_param)
                model.fit(train_X, train_y)
                train_labels['yhat'] = model.predict(train_X)
                validation_labels['yhat'] = model.predict(validation_X)
            
            # Evaluate
            train_labels = train_labels.groupby('file_name').mean()
            validation_labels = validation_labels.groupby('file_name').mean()
            all_validation_labels.append(validation_labels.reset_index())
            
            train_y = train_labels['y'].tolist()
            train_yhat = train_labels['yhat'].tolist()
            validation_y = validation_labels['y'].tolist()
            validation_yhat = validation_labels['yhat'].tolist()
            
            # list of values
            all_labels.extend(validation_y)
            all_predictions.extend(validation_yhat)
            
            train_mse = mean_squared_error(train_y, train_yhat)
            train_mae = mean_absolute_error(train_y, train_yhat)
            train_r2 = r2_score(train_y, train_yhat)
            train_corr = pearsonr(train_y, train_yhat)[0]
            
            validation_mse = mean_squared_error(validation_y, validation_yhat)
            validation_mae = mean_absolute_error(validation_y, validation_yhat)
            validation_r2 = r2_score(validation_y, validation_yhat)
            validation_corr, p_value = pearsonr(validation_y, validation_yhat)
            
            train_mses.append(train_mse)
            train_maes.append(train_mae)
            train_r2s.append(train_r2)
            train_corrs.append(train_corr)
            
            validation_mses.append(validation_mse)
            validation_maes.append(validation_mae)
            validation_r2s.append(validation_r2)
            validation_corrs.append(validation_corr)
            validation_corr_pvalues.append(p_value)
            
            if self.verbose:
                print(f'Fold: {index+1}, TrainMSE: {np.round(train_mse, 3)}, TrainMAE: {np.round(train_mae, 3)}, ValMSE: {np.round(validation_mse, 3)}, ValMAE: {np.round(validation_mae, 3)}, ValR2: {np.round(validation_r2, 3)}, ValCorr: {np.round(validation_corr, 3)}, ValPValue: {np.round(p_value, 3)}')
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Save file_names with y and yhat for analyzing results for each book
        _ = self._concat_and_save_examples(all_validation_labels)
        
        mean_train_mse = np.mean(train_mses)
        mean_train_rmse = np.mean([sqrt(x) for x in train_mses])
        mean_train_mae = np.mean(train_maes)
        mean_train_r2 = np.mean(train_r2s)
        mean_train_corr = np.mean(train_corrs)
        
        mean_validation_mse = np.mean(validation_mses)
        mean_validation_rmse = np.mean([sqrt(x) for x in validation_mses])
        mean_validation_mae = np.mean(validation_maes)
        mean_validation_r2 = np.mean(validation_r2s)
        mean_validation_corr = np.mean(validation_corrs)
        mean_p_value = self._get_pvalue(validation_corr_pvalues)
        
        if self.verbose:
            print(f'''TrainMSE: {np.round(mean_train_mse, 3)}, 
                TrainRMSE: {np.round(mean_train_rmse, 3)}, 
                TrainMAE: {np.round(mean_train_mae, 3)}, 
                TrainR2: {np.round(mean_train_r2, 3)}, 
                TrainCorr: {np.round(mean_train_corr, 3)}, 
                ValMSE: {np.round(mean_validation_mse, 3)}, 
                ValRMSE: {np.round(mean_validation_rmse, 3)}, 
                ValMAE: {np.round(mean_validation_mae, 3)}, 
                ValR2: {np.round(mean_validation_r2, 3)}, 
                ValCorr: {np.round(mean_validation_corr, 3)}, 
                ValCorrPValue: {np.round(mean_p_value, 3)}''')
   
            if self.language == 'eng':
                color = 'm'
            else:
                color = 'teal'
            plt.figure(figsize=(4,4))
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            #plt.xlim([min(all_labels) - 0.01, max(all_labels) + 0.01])
            #plt.ylim([min(all_predictions) - 0.01, max(all_predictions) + 0.01])
            plt.xticks(np.arange(round(min(all_labels),2) - 0.01, round(max(all_labels),2) + 0.01, 0.03))
            plt.scatter(x=all_labels, y=all_predictions, s=6, c=color)
            plt.xlabel('True Scores', fontsize=15)
            plt.ylabel('Predicted Scores', fontsize=15)
            plt.savefig(f'{self.results_dir}{self.model_info_string}.png', dpi=400, bbox_inches='tight')
            plt.show();

        returned_values = [mean_train_mse, 
            mean_train_rmse, 
            mean_train_mae, 
            mean_train_r2, 
            mean_train_corr, 
            mean_validation_mse, 
            mean_validation_rmse, 
            mean_validation_mae, 
            mean_validation_r2, 
            mean_validation_corr, 
            mean_p_value]
        return [round(x, 3) for x in returned_values]


class TwoclassClassification(Regression):
    ''' Classify into reviewed/not reviewed.'''
    def __init__(self, results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True):
        super().__init__(results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True)

    def _check_class_specific_assertions(self):
        assert self.model in ['svc', 'xgboost']
        assert self.features_string in ['book', 'baac']
        assert self.labels_string in ['twoclass', 'library']
        
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
    
    def _get_accuracy(self, df):
        return [accuracy_score(df['y'], df['yhat']), balanced_accuracy_score(df['y'], df['yhat'])] 

    def _make_crosstabs(self, all_validation_labels):
        crosstab = pd.crosstab(all_validation_labels['y'], all_validation_labels['yhat'], rownames=['True'], colnames=['Predicted'], margins=True)
        crosstab.to_csv(f'{self.results_dir}crosstab_{self.model_info_string}.csv', index=True)
        print('--------------------------\nCrosstab\n', crosstab, '\n--------------------------')
                    
    def run(self):
        train_accs = []
        train_balanced_accs = []
        validation_accs = []
        validation_balanced_accs = []
        df = self.df
        df = self._combine_df_labels(df)
        file_names_split_stratified = CVFolds(df=df, n_folds=5, seed=1, stratified=True, return_indices=False).split()
        all_validation_labels = []

        for index, split in enumerate(file_names_split_stratified):
            train_df, train_X, train_y, validation_X, train_labels, validation_labels = self._prepare_dfs(split, df)
            
            # Train model
            if self.model == 'xgboost':
                best_parameters = self._get_model(self.model_param, train_X, train_y, train_df, task_type='binary_classification')
                class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=train_y.astype(int).tolist())))
                dtrain = xgboost.DMatrix(train_X, label=train_y.astype(int), weight=[class_weights[int(i)] for i in train_y])
                num_boost_round = best_parameters['num_boost_round']
                best_parameters.pop('nested_cv_score')
                best_parameters.pop('num_boost_round')
                model = xgboost.train(best_parameters,
                                      dtrain,
                                      num_boost_round=num_boost_round,
                                      verbose_eval=False)
                train_labels['yhat'] = model.predict(xgboost.DMatrix(train_X))
                validation_labels['yhat'] = model.predict(xgboost.DMatrix(validation_X))
            else:
                model = self._get_model(self.model_param)
                model.fit(train_X, train_y)
                train_labels['yhat'] = model.predict(train_X)
                validation_labels['yhat'] = model.predict(validation_X)

            # list of dfs
            all_validation_labels.append(validation_labels)

            # Evaluate
            train_acc, train_balanced_acc = self._get_accuracy(train_labels)
            validation_acc, validation_balanced_acc = self._get_accuracy(validation_labels)
            
            train_accs.append(train_acc)
            train_balanced_accs.append(train_balanced_acc)
            validation_accs.append(validation_acc)
            validation_balanced_accs.append(validation_balanced_acc)
        
        # Save y and y_pred for examples
        all_validation_labels = self._concat_and_save_examples(all_validation_labels)

        self._make_crosstabs(all_validation_labels)

        mean_train_acc = statistics.mean(train_accs)
        mean_train_balanced_acc = statistics.mean(train_balanced_accs)
        mean_validation_acc = statistics.mean(validation_accs)
        mean_validation_balanced_acc = statistics.mean(validation_balanced_accs)

        return [round(mean_train_acc, 3), round(mean_train_balanced_acc, 3), round(mean_validation_acc, 3), round(mean_validation_balanced_acc, 3)]

class MulticlassClassification(TwoclassClassification):
    '''Classify into not reviewed/negative/not classified/positive.'''
    def __init__(self, results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True):
        super().__init__(results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True)

    def _check_class_specific_assertions(self):
        assert self.model in ['svc', 'xgboost']
        assert self.features_string in ['book', 'baac']
        assert self.labels_string in ['multiclass']
                
    def _get_f1_score(self, df):
        score = f1_score(df['y'], df['yhat'], average='macro')
        return score
        
    def run(self):
        train_f1s = []
        validation_f1s = []

        df = self.df
        df = self._combine_df_labels(df)
        file_names_split_stratified = CVFolds(df=df, n_folds=5, seed=1, stratified=True, return_indices=False).split()
        all_validation_labels = []

        for index, split in enumerate(file_names_split_stratified):
            train_df, train_X, train_y, validation_X, train_labels, validation_labels = self._prepare_dfs(split, df)
    
            if self.model == 'xgboost':
                best_parameters = self._get_model(self.model_param, train_X, train_y, train_df, task_type='multiclass_classification')
                class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1, 2, 3], y=train_y.astype(int).tolist())))
                dtrain = xgboost.DMatrix(train_X, label=train_y.astype(int), weight=[class_weights[int(i)] for i in train_y])
                num_boost_round = best_parameters['num_boost_round']
                best_parameters.pop('nested_cv_score')
                best_parameters.pop('num_boost_round')
                model = xgboost.train(best_parameters,
                                      dtrain,
                                      num_boost_round=num_boost_round,
                                      verbose_eval=False)
                train_labels['yhat'] = model.predict(xgboost.DMatrix(train_X))
                validation_labels['yhat'] = model.predict(xgboost.DMatrix(validation_X))
            else:
                model = self._get_model(self.model_param)
                model.fit(train_X, train_y)
                train_labels['yhat'] = model.predict(train_X)
                validation_labels['yhat'] = model.predict(validation_X)
            all_validation_labels.append(validation_labels)

            train_f1 = self._get_f1_score(train_labels)
            train_f1s.append(train_f1)
            validation_f1 = self._get_f1_score(validation_labels)           
            validation_f1s.append(validation_f1)
            if self.verbose:
                print(f'Fold: {index+1}, TrainF1: {np.round(train_f1, 3)}, ValF1: {np.round(validation_f1, 3)}')
        
        # Save y and yhat for examples
        all_validation_labels = self._concat_and_save_examples(all_validation_labels)

        self._make_crosstabs(all_validation_labels)

        mean_train_f1 = statistics.mean(train_f1s)
        mean_validation_f1 = statistics.mean(validation_f1s)
        
        if self.verbose:
            print(f'''TrainF1: {np.round(mean_train_f1, 3)}, ValidationF1: {np.round(mean_validation_f1, 3)}''')
            print('\n---------------------------------------------------\n')
        return [round(mean_train_f1, 3), round(mean_validation_f1, 3)]
