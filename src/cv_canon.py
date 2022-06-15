# %%
%load_ext autoreload
%autoreload 2
language = 'eng'

import sys
sys.path.insert(0, '../src/')
import numpy as np
import pandas as pd
import os

features_dir = f'../data/features_30/{language}/'
results_dir = f'../data/results_canon/{language}/'
sentiment_dir = '../data/labels_sentiment/'
canonization_labels_dir = '../data/labels_canon/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# %%

    def __init__(self, language, features, drop_columns, dimensionality_reduction, model_param, model, verbose, params_to_use):
        assert isinstance(drop_columns, list)
        for i in drop_columns:
            assert isinstance(i, str)
        assert (dimensionality_reduction in ['k_best_f_reg_0_10', 'k_best_mutual_info_0_10', 'ss_pca_0_95', 'rfe']) or (dimensionality_reduction is None)
        self._check_class_specific_assertions()
        
        self.language = language
        self.features = features
        self.labels = labels
        self.drop_columns = drop_columns
        self.dimensionality_reduction = dimensionality_reduction
        self.model_param = model_param
        self.model = model
        self.verbose = verbose
        self.params_to_use = params_to_use
        self.best_features = []

        if self.features == 'book':
            self.df = deepcopy(book_df)
        elif self.features == 'chunk':
            self.df = deepcopy(chunk_df)
        elif self.features == 'chunk_and_copied_book':
            self.df = deepcopy(chunk_and_copied_book_df)
        elif self.features == 'book_and_averaged_chunk':
            self.df = deepcopy(book_and_averaged_chunk_df)

        columns_before_drop = set(self.df.columns)
        if self.drop_columns:
            self.df = self.df[[column for column in self.df.columns if not self._drop_column(column)]].reset_index(drop=True)
        columns_after_drop = set(self.df.columns)
        if self.verbose:
            print(f'Dropped {len(columns_before_drop - columns_after_drop)} columns.')
            
    def _check_class_specific_assertions(self):
        assert model in ['xgboost', 'svr', 'lasso']
        assert features in ['book', 'chunk', 'book_and_averaged_chunk', 'chunk_and_copied_book']

    def _drop_column(self, column):
        for string in self.drop_columns:
            if string in column:
                return True
        return False
    
    def _custom_pca(self, train_X):
        for i in range(5, train_X.shape[1], int((train_X.shape[1] - 5) / 10)):
            pca = PCA(n_components=i)
            new_train_X = pca.fit_transform(train_X)
            if pca.explained_variance_ratio_.sum() >= 0.95:
                break
        return new_train_X, pca

    def _select_features(self, train_X, train_y, validation_X, train_file_names):
        if self.dimensionality_reduction == 'ss_pca_0_95':
            ss = StandardScaler()
            train_X = ss.fit_transform(train_X)
            validation_X = ss.transform(validation_X)
            train_X, pca = self._custom_pca(train_X)
            validation_X = pca.transform(validation_X)
        elif self.dimensionality_reduction == 'k_best_f_reg_0_10':
            #Find best featues
            k_best = SelectKBest(f_regression, k=np.minimum(int(0.10 * train_X.shape[0]), train_X.shape[1]))
            k_best = k_best.fit(train_X, train_y)
            mask = k_best.get_support(indices=True)
            selected_features_df = train_X.iloc[:,mask]
            self.best_features.extend(selected_features_df.columns.tolist())
            
            print('trainX before feature selection', train_X.shape)
            train_X = k_best.transform(train_X)
            print('trainX after feature selection', train_X.shape)
            validation_X = k_best.transform(validation_X)

        elif self.dimensionality_reduction == 'k_best_mutual_info_0_10':
            k_best = SelectKBest(mutual_info_regression, k=np.minimum(int(0.10 * train_X.shape[0]), train_X.shape[1]))
            train_X = k_best.fit_transform(train_X, train_y)
            validation_X = k_best.transform(validation_X)
        elif self.dimensionality_reduction is None:
            pass
        print('Feature Selection Done')
        return train_X, validation_X
    
    def _impute(self, train_X, validation_X):
        imputer = KNNImputer()
        train_X = imputer.fit_transform(train_X)
        validation_X = imputer.transform(validation_X)
        return train_X, validation_X
    
    def _get_model(self, model_param, train_X=None, train_y=None, train_file_names=None, task_type=None):
        if self.model == 'xgboost':
            if task_type == 'binary_classification':
                is_classification = True
                class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=train_y.astype(int).tolist())))
            elif task_type == 'multiclass_classification':
                is_classification = True
                class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1, 2, 3], y=train_y.astype(int).tolist())))
            elif task_type == 'regression':
                is_classification = False
            else:
                raise Exception('Not a valid task_type')
            
            def feval(preds, train_data):
                labels = train_data.get_label()
                if is_classification:
                    labels = labels.astype(int)
                    preds = preds.argmax(axis=1).astype(int)
                    if task_type == 'binary_classification':
                        return 'acc', accuracy_score(labels, preds)
                    elif task_type == 'multiclass_classification':
                        return 'f1', f1_score(labels, preds, average='macro')
                else:
                    return 'rmse', np.sqrt(mean_squared_error(labels, preds))
            
            if is_classification:
                dtrain = xgboost.DMatrix(train_X, label=train_y.astype(int), weight=[class_weights[int(i)] for i in train_y])
            else:
                dtrain = xgboost.DMatrix(train_X, label=train_y)
            results = []
            df = np.hstack((train_file_names, train_X))
            df = pd.DataFrame(df, columns=['file_name'] + [f'col_{i}' for i in range(train_X.shape[1])])
            for max_depth in [2, 4, 6, 8]:
                for learning_rate in [None, 0.01, 0.033, 0.1]:
                    for colsample_bytree in [0.33, 0.60, 0.75]:
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
                                        num_boost_round=99999,
                                        seed=42,
                                        nfold=5,
                                        folds=AuthorSplit(df, 5, seed=8, return_indices=True).split(),
                                        feval=feval,
                                        maximize=is_classification, # if classification, maximize f1/acc score.
                                        early_stopping_rounds=10,
                                        verbose_eval=False)

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
            best_parameters = sorted(results, key=lambda x: x['nested_cv_score'], reverse=is_classification)[0]
            return best_parameters
        elif self.model == 'svr':
            return SVR(C=model_param)
        elif self.model == 'lasso':
            return Lasso(alpha=model_param)
        elif self.model == 'svc':
            return SVC(C=model_param, class_weight='balanced')
        
    
    def _get_pvalue(self, validation_corr_pvalues):
        # Harmonic mean p-value
        denominator = sum([1/x for x in validation_corr_pvalues])
        mean_p_value = len(validation_corr_pvalues)/denominator
        return mean_p_value
    
    def _combine_df_labels(self, df):
        #Average of sentiscores per book
        df = df.merge(right=self.labels, on='file_name', how='inner', validate='many_to_one')
        return df
    
    def run(self):
        all_predictions = []
        all_labels = []

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
        file_names_split = AuthorSplit(df, 5, seed=2, return_indices=False).split() ## 10 folds
        all_validation_books = []

        for index, split in enumerate(file_names_split):
            train_df = df[~df['file_name'].isin(split)]
            validation_df = df[df['file_name'].isin(split)]
            
            train_X = train_df.drop(columns=['y', 'file_name'])
            train_y = train_df['y']
            validation_X = validation_df.drop(columns=['y', 'file_name'])
            validation_y = validation_df['y']
            """
            train_X = train_df.drop(columns=['y', 'file_name']).values
            train_y = train_df['y'].values.ravel()
            validation_X = validation_df.drop(columns=['y', 'file_name']).values
            validation_y = validation_df['y'].values.ravel()
            """
            #Impute missing values if df contains NaNs
            #train_X, validation_X = self._impute(train_X, validation_X)
            #if self.verbose:
            #    print(f'train_X.shape before {self.dimensionality_reduction}: {train_X.shape}, validation_X.shape before {self.dimensionality_reduction}: {validation_X.shape}')
            train_X, validation_X = self._select_features(train_X, train_y, validation_X, train_file_names=train_df['file_name'].values.reshape(-1, 1))
            #if self.verbose:
            #    print(f'train_X.shape after {self.dimensionality_reduction}: {train_X.shape}, validation_X.shape after {self.dimensionality_reduction}: {validation_X.shape}')
            if self.model == 'xgboost':
                train_file_names = train_df['file_name'].values.reshape(-1, 1)
                best_parameters = self._get_model(self.model_param, train_X, train_y, train_file_names, task_type='regression')
                dtrain = xgboost.DMatrix(train_X, label=train_y)
                num_boost_round = best_parameters['num_boost_round']
                best_parameters.pop('nested_cv_score')
                best_parameters.pop('num_boost_round')
                model = xgboost.train(best_parameters,
                                      dtrain,
                                      num_boost_round=num_boost_round,
                                      verbose_eval=False)
            else:
                model = self._get_model(self.model_param)
                model.fit(train_X, train_y)
            
            train_books = deepcopy(train_df[['file_name', 'y']])
            validation_books = deepcopy(validation_df[['file_name', 'y']])
            
            if self.model == 'xgboost':
                train_books['yhat'] = model.predict(xgboost.DMatrix(train_X))
                validation_books['yhat'] = model.predict(xgboost.DMatrix(validation_X))
                
                print('train preds:', model.predict(xgboost.DMatrix(train_X)))
                print('validation preds:', model.predict(xgboost.DMatrix(validation_X)))
            else:
                train_books['yhat'] = model.predict(train_X)
                validation_books['yhat'] = model.predict(validation_X)
            
            train_books = train_books.groupby('file_name').mean()
            validation_books = validation_books.groupby('file_name').mean()
            all_validation_books.append(validation_books.reset_index())
            
            train_y = train_books['y'].tolist()
            train_yhat = train_books['yhat'].tolist()
            validation_y = validation_books['y'].tolist()
            validation_yhat = validation_books['yhat'].tolist()
            
            all_labels.extend(validation_y)
            all_predictions.extend(validation_yhat)
            
            train_mse = mean_squared_error(train_y, train_yhat)
            train_mae = mean_absolute_error(train_y, train_yhat)
            train_r2 = r2_score(train_y, train_yhat)
            train_corr = pearsonr(train_y, train_yhat)[0]
            
            validation_mse = mean_squared_error(validation_y, validation_yhat)
            validation_mae = mean_absolute_error(validation_y, validation_yhat)
            validation_r2 = r2_score(validation_y, validation_yhat)
            validation_corr = pearsonr(validation_y, validation_yhat)[0]
            p_value = pearsonr(validation_y, validation_yhat)[1]
            
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
                print(f'Fold: {index+1}, TrainMSE: {np.round(train_mse, 3)}, TrainMAE: {np.round(train_mae, 3)}, ValMSE: {np.round(validation_mse, 3)},'
                    f'ValMAE: {np.round(validation_mae, 3)}, ValR2: {np.round(validation_r2, 3)}, ValCorr: {np.round(validation_corr, 3)}')
        print('loop finished')
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Save y and y_pred for examples
        model_info_string = f'{self.language}-{self.model}-{self.dimensionality_reduction}-{self.features}-pram{self.model_param}-{self.params_to_use}'
        pd.concat(all_validation_books).to_csv(f'{results_dir}y_yhat-{model_info_string}.csv', index=False)
       
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
            print(f'TrainMSE: {np.round(mean_train_mse, 3)}, TrainRMSE: {np.round(mean_train_rmse, 3)}, TrainMAE: {np.round(mean_train_mae, 3)}, TrainR2: {np.round(mean_train_r2, 3)},'
                    f'TrainCorr: {np.round(mean_train_corr, 3)}, ValMSE: {np.round(mean_validation_mse, 3)}, ValRMSE: {np.round(mean_validation_rmse, 3)}, ValMAE: {np.round(mean_validation_mae, 3)},'
                    f'ValR2: {np.round(mean_validation_r2, 3)}, ValCorr: {np.round(mean_validation_corr, 3)}, ValCorrPValue: {np.round(mean_p_value, 3)}')
            plt.figure(figsize=(4,4))
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlim([0,1])
            plt.ylim([0,1])

            plt.scatter(all_labels, all_predictions, s=6)
            plt.xlabel('Canonization Scores', fontsize=20)
            plt.ylabel('Predicted Scores', fontsize=20)
            plt.savefig(f'{results_dir}{model_info_string}.png', dpi=400, bbox_inches='tight')   
            plt.show();
            print('\n---------------------------------------------------\n')

        best_features = dict(Counter(self.best_features))
        
        return (
            mean_train_mse, 
            mean_train_rmse, 
            mean_train_mae, 
            mean_train_r2, 
            mean_train_corr, 
            mean_validation_mse, 
            mean_validation_rmse, 
            mean_validation_mae, 
            mean_validation_r2, 
            mean_validation_corr, 
            mean_p_value, 
            best_features)

# %%
"""
Parameter combinations
"""
drop_columns_list = [
    ['average_sentence_embedding', '100_most_common_', 'doc2vec_chunk_embedding'],
    ['average_sentence_embedding', '100_most_common_', 'doc2vec_chunk_embedding', 'pos'],
    ]
if language == 'eng':
    drop_columns_list.extend([
        ['average_sentence_embedding', '100_most_common_', 'doc2vec_chunk_embedding', '->'], 
        ['average_sentence_embedding', '100_most_common_', 'doc2vec_chunk_embedding', '->', 'pos']
    ])
    
models = ['svr', 'lasso', 'xgboost', 'svc']
model_params = {'svr': [1], 'lasso': [1, 4], 'xgboost': [None], 'svc': [0.1, 1, 10, 100, 1000, 10000]} 
dimensionality_reduction = ['ss_pca_0_95', 'k_best_f_reg_0_10', 'k_best_mutual_info_0_10', 'rfe', [None]]
features = ['book', 'chunk', 'book_and_averaged_chunk', 'chunk_and_copied_book']

regression_params = {'model': models[2], 'dimensionality_reduction': dimensionality_reduction[-1], 'features': [features[0]]}
testing_params = {'model': models[0], 'dimensionality_reduction': dimensionality_reduction[1], 'features': [features[0]]}

# %%
"""
Run Regression
"""
results = []
params_to_use = 'testing' 
if params_to_use == 'regression':
    param_dict = regression_params
elif params_to_use == 'testing':
    param_dict = testing_params
elif params_to_use == 'multiclass':
    param_dict = multiclass_params
elif params_to_use == 'full_cv':
    param_dict = full_cv_params

book_df = pd.read_csv(f'{features_dir}book_df.csv')
print(book_df.isnull().values.any())
# book_and_averaged_chunk_df = pd.read_csv(f'{features_dir}book_and_averaged_chunk_df.csv')
# chunk_df = pd.read_csv(f'{features_dir}chunk_df.csv')
# chunk_and_copied_book_df = pd.read_csv(f'{features_dir}chunk_and_copied_book_df.csv')

book_df = book_df.loc[book_df['file_name'] != 'Defoe_Daniel_Roxana_1724'] ########################################
# book_and_averaged_chunk_df = book_and_averaged_chunk_df.loc[book_and_averaged_chunk_df['file_name'] != 'Defoe_Daniel_Roxana_1724']
# chunk_df = chunk_df.loc[chunk_df['file_name'] != 'Defoe_Daniel_Roxana_1724']
# chunk_and_copied_book_df = chunk_and_copied_book_df.loc[chunk_and_copied_book_df['file_name'] != 'Defoe_Daniel_Roxana_1724']


for language in [language]:
    for model in [param_dict['model']]:
        model_param = model_params[model]
        for model_param in model_param:
            for dimensionality_reduction in [param_dict['dimensionality_reduction']]:
                for features in param_dict['features']:
                    for drop_columns in [drop_columns_list[3]]:
                        print(params_to_use, language, model, features, drop_columns, dimensionality_reduction, 'param=', model_param)
                        try:
                            experiment = Regression(
                                language=language,
                                features=features,
                                drop_columns=drop_columns,
                                dimensionality_reduction = dimensionality_reduction,
                                model_param=model_param,
                                model=model,
                                verbose=True,
                                params_to_use = params_to_use)
                            returned_values = experiment.run()
                            results.append((language, model, features, drop_columns, dimensionality_reduction, model_param) + returned_values)
                        except Exception as e:
                            print(f'Error in {language}, {model}, {features}, {drop_columns}, {dimensionality_reduction}')
                            print(e)
                            raise e
    results_df = pd.DataFrame(results, columns=['language', 'model', 'features', 'drop_columns', 
    'dimensionality_reduction', 'model_param', 'mean_train_mse', 'mean_train_rmse', 
    'mean_train_mae', 'mean_train_r2', 'mean_train_corr', 'mean_validation_mse', 'mean_validation_rmse',
    'mean_validation_mae', 'mean_validation_r2', 'mean_validation_corr', 'mean_p_value', 'best_features'])
    results_df.to_csv(f'{results_dir}results-{language}-{params_to_use}.csv', index=False)

# %%
# Get distances based on best features
best_features = list(results_df.loc[0, 'best_features'].keys())
reduced_df = book_df[['file_name'] + best_features]

from scipy.spatial.distance import squareform, pdist
matrix = pd.DataFrame(squareform(pdist(reduced_df.iloc[:,1:], metric='cosine')), index = reduced_df['file_name'], columns = reduced_df['file_name'])
matrix

# %%
# Arbitrary threshold for splitting books into canonized, not canonized
discrete_labels = labels.copy()
discrete_labels['y'] = (discrete_labels['y']>0.5).astype(int)

discrete_labels['y'].hist(bins=100)

# %%
discrete_labels = discrete_labels[discrete_labels['file_name'].isin(matrix.columns.tolist())]
discrete_labels['file_name']==matrix.index


