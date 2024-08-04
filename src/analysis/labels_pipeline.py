
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    log_loss, 
    matthews_corrcoef, 
    roc_curve
)

import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from xgboost import XGBClassifier

import sys
sys.path.append("..")
from cluster.network import NXNetwork
from cluster.combinations import InfoHandler
from utils import DataHandler

class LabelPredict(DataHandler):
    def __init__(self, language, by_author=False, attr='author', test=False, simmx_path=None):
        super().__init__(language=language, output_dir='label_predict', by_author=by_author, data_type='csv')
        self.ih = InfoHandler(language=self.language, add_color=False, cmode=None, by_author=self.by_author)
        self.attr = attr
        self.test = test
        self.metadf = self.ih.metadf[self.attr]
        self.data_split_path = os.path.join(self.output_dir, f'upsampled_splits_{self.attr}.pkl')
        self.simmx_path = simmx_path
        print('nr texts', self.nr_texts)

    def is_symmetric(self, df):
        # Check if the DataFrame is square
        assert df.shape[0] == df.shape[1]
        # Check if the index and column names are the same
        assert df.index.tolist() == df.columns.tolist()
        # Check if the DataFrame is symmetric
        return (df == df.T).all().all()

    def get_simmx(self):
        if self.simmx_path is None:
            self.simmx_path = os.path.join(self.data_dir, 'similarity', self.language, 'simmxs/d2v-full.csv')
            print('simmx path', self.simmx_path)
            df = pd.read_csv(self.simmx_path, index_col=0)
            # Check if the first column has a name
            if df.index.name is None:
                df.index.name = 'file_name'
        elif isinstance(self.simmx_path, str) and os.path.exists(self.simmx_path) and self.simmx_path.endswith('.pkl'):
            network = NXNetwork(self.language, path=self.simmx_path)
            df = network.mx.mx
        else:
            raise ValueError(f'path cannot be loaded: {self.simmx_path}')
        return df

    def get_long_format_df(self):
        df = self.get_simmx()
        is_symmetric = self.is_symmetric(df)
        df = df.reset_index()
        assert 'file_name' in df.columns
        df = df.rename(columns={'file_name': 'left'})
        df = df.melt(id_vars='left', var_name='right', value_name='weight')

        # Filter out self-pairs
        df = df[df['left'] != df['right']]
        # Add attributes
        df = df.merge(self.metadf, left_on='left', right_index=True)
        df = df.rename(columns={self.attr: 'attr_left'})
        df = df.merge(self.metadf, left_on='right', right_index=True)
        df = df.rename(columns={self.attr: 'attr_right'})
        df = df.reset_index(drop=True)

        if is_symmetric:
            # Drop symmetric duplicated pairs by keeping only (i, j) where i < j
            df['left_right'] = df.apply(lambda row: tuple(sorted([row['left'], row['right']])), axis=1)
            df = df.drop_duplicates(subset='left_right').drop(columns='left_right')


        if is_symmetric:
            assert len(df) == (self.nr_texts * (self.nr_texts -1))/2
        else:
            assert len(df) == (self.nr_texts * (self.nr_texts -1))

        return df, is_symmetric


    def load_data(self):
        df, is_symmetric = self.get_long_format_df()
        df['equal'] = df.apply(lambda row: row['attr_left'] == row['attr_right'], axis=1)
        if self.attr == 'gender':
            df = df[~df['left'].str.contains('Stevenson-Grift_Robert-Louis-Fanny-van-de_The-Dynamiter_1885|Anonymous', na=False) &
                            ~df['right'].str.contains('Stevenson-Grift_Robert-Louis-Fanny-van-de_The-Dynamiter_1885|Anonymous', na=False)]

        value_counts = df['equal'].value_counts()
        print(value_counts)
        print(f'The share of True by the share of False is: {value_counts[True] / value_counts[False]}')
        # Define features and target
        X = df[['weight']]
        y = df['equal'].astype(int)
        return df, X, y
    
    
    def data_exploration(self):
        df, X, y = self.load_data()
        # Assuming df is your DataFrame
        # Step 1: Visualization
        sns.scatterplot(x='weight', y='equal', data=df)
        plt.xlabel('Weight')
        plt.ylabel('Equal')
        plt.title('Scatter Plot of Weight vs Equal')
        plt.show()

        # Step 2: Summary Statistics
        mean_weight_true = df[df['equal']]['weight'].mean()
        mean_weight_false = df[~df['equal']]['weight'].mean()

        print(f'Mean weight when equal is True: {mean_weight_true}')
        print(f'Mean weight when equal is False: {mean_weight_false}')

        # # Step 3: Correlation Analysis
        # corr, _ = pointbiserialr(df['weight'], df['equal'])
        # print(f'Point-Biserial Correlation: {corr}')


    def get_upsampled_splits(self):
        from imblearn.over_sampling import SMOTE # only installed in conda env networkclone
        from sklearn.model_selection import StratifiedKFold

        df, X, y = self.load_data()
        
        # Split data into training and test sets (15% for final testing)
        X_train_full, X_final_test, y_train_full, y_final_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=4)

        folds = {}
        counter = 0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
        for train_index, test_index in skf.split(X_train_full, y_train_full):

            X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
            y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]

            if self.test:
                # Small test sample
                sample_size = int(len(X_train) * 0.01)
                X_train = X_train.sample(n=sample_size, random_state=42)
                y_train = y_train.loc[X_train.index]


            # Address the class imbalance using SMOTE
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            folds[counter] = [X_train_res, y_train_res, X_train, X_test, y_train, y_test]
            counter += 1

        # Save the final test set separately
        folds['final_test'] = [X_final_test, y_final_test]

        with open(self.data_split_path, 'wb') as file:
            pickle.dump(folds, file)



    def classifier_pipeline(self):
        with open(self.data_split_path, 'rb') as file:
            folds = pickle.load(file)

        classifiers = {
            'Logistic Regression': LogisticRegression(n_jobs=-1),
            'Random Forest': RandomForestClassifier(n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Support Vector Machine': SVC(probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
            'Naive Bayes': GaussianNB(),
            'Extra Trees': ExtraTreesClassifier(n_jobs=-1),
            'Bagging': BaggingClassifier(n_jobs=-1),
            'XGBoost': XGBClassifier(eval_metric='logloss', n_jobs=-1)
        }

        pipelines = {}
        for name, classifier in classifiers.items():
            pipelines[name] = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('classifier', classifier)
            ])

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Log Loss', 'MCC']
        results = {name: {metric: [] for metric in metrics} for name in pipelines}

        X_final_test, y_final_test = folds.pop('final_test')

        print('Starting cross-validation...')
        for fold_key, (X_train_res, y_train_res, X_train, X_test, y_train, y_test) in folds.items():
            print(f'\n--------- Fold: {fold_key} ----------------\n')

            for name, pipeline in pipelines.items():
                print(f'Training and evaluating {name} on fold {fold_key}...')
                pipeline.fit(X_train_res, y_train_res)
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else None
                
                results[name]['Accuracy'].append(accuracy_score(y_test, y_pred))
                results[name]['Precision'].append(precision_score(y_test, y_pred, average='binary'))
                results[name]['Recall'].append(recall_score(y_test, y_pred, average='binary'))
                results[name]['F1 Score'].append(f1_score(y_test, y_pred, average='binary'))
                if y_pred_proba is not None:
                    results[name]['ROC AUC'].append(roc_auc_score(y_test, y_pred_proba))
                    results[name]['Log Loss'].append(log_loss(y_test, y_pred_proba))
                results[name]['MCC'].append(matthews_corrcoef(y_test, y_pred))

                print(f'{name} - Accuracy: {accuracy_score(y_test, y_pred):.3f}, '
                    f'Precision: {precision_score(y_test, y_pred, average="binary"):.3f}, '
                    f'Recall: {recall_score(y_test, y_pred, average="binary"):.3f}, '
                    f'F1 Score: {f1_score(y_test, y_pred, average="binary"):.3f}')
                print('Confusion Matrix:')
                print(confusion_matrix(y_test, y_pred))
                print('Classification report')
                print(classification_report(y_true=y_test, y_pred=y_pred))

        print('\nFinal test set evaluation...')
        final_results = {name: {} for name in pipelines}
        for name, pipeline in pipelines.items():
            print(f'Retraining {name} on the full training data...')
            X_train_full = pd.concat([folds[fold][2] for fold in folds])
            y_train_full = pd.concat([folds[fold][4] for fold in folds])

            pipeline.fit(X_train_full, y_train_full)
            y_final_pred = pipeline.predict(X_final_test)
            y_final_pred_proba = pipeline.predict_proba(X_final_test)[:, 1] if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else None

            final_results[name] = {
                'Accuracy': accuracy_score(y_final_test, y_final_pred),
                'Precision': precision_score(y_final_test, y_final_pred, average='binary'),
                'Recall': recall_score(y_final_test, y_final_pred, average='binary'),
                'F1 Score': f1_score(y_final_test, y_final_pred, average='binary'),
                'ROC AUC': roc_auc_score(y_final_test, y_final_pred_proba) if y_final_pred_proba is not None else None,
                'Log Loss': log_loss(y_final_test, y_final_pred_proba) if y_final_pred_proba is not None else None,
                'MCC': matthews_corrcoef(y_final_test, y_final_pred)
            }

            print(f'{name} on final test set - Accuracy: {accuracy_score(y_final_test, y_final_pred):.3f}, '
                f'Precision: {precision_score(y_final_test, y_final_pred, average="binary"):.3f}, '
                f'Recall: {recall_score(y_final_test, y_final_pred, average="binary"):.3f}, '
                f'F1 Score: {f1_score(y_final_test, y_final_pred, average="binary"):.3f}')
            print('Confusion Matrix:')
            print(confusion_matrix(y_final_test, y_final_pred))
            print('Classification report')
            print(classification_report(y_true=y_test, y_pred=y_pred))

        results_aggregated = {}
        for name, metrics in results.items():
            results_aggregated[name] = {metric: metrics[metric] for metric in metrics}
            for metric in metrics:
                results_aggregated[name][f'{metric} Mean'] = np.mean(metrics[metric])
                results_aggregated[name][f'{metric} Std'] = np.std(metrics[metric])

        results_df = pd.DataFrame(results_aggregated).T
        results_df.columns = [f'{metric} (Fold Metrics)' for metric in metrics] + [f'{metric} Mean' for metric in metrics] + [f'{metric} Std' for metric in metrics]
        results_df = results_df.round(3)

        final_test_df = pd.DataFrame(final_results).T
        final_test_df.columns = [f'Final Test {metric}' for metric in metrics]
        
        results_combined_df = results_df.join(final_test_df)
        print('\nFinal combined results:')
        print(results_combined_df)
        
        self.save_data(file_name=f'pipeline_results_{self.attr}.csv', data=results_combined_df)



class LabelPredictCont(LabelPredict):
    def __init__(self, language, by_author=False, attr=None, test=False):
        super().__init__(language=language, by_author=by_author, attr=attr)
        self.test = test


    def load_data(self):
        df, is_symmetric = self.get_long_format_df()
        df['target'] = df['attr_right'] - df['attr_left']
        if is_symmetric:
            # If symmetric, it should not matter which value is left or right in the difference calculation
            df['target'] = abs(df['target'])

        X = df[['weight']]
        y = df['target']
        return df, X, y

    
    def data_exploration(self):
        df, X, y = self.load_data()
        # Data exploration
        print(df.describe())

        # Scatter plot to visualize the relationship
        sns.scatterplot(x='weight', y='target', data=df)
        plt.title('Scatter plot of independent_var vs dependent_var')
        plt.xlabel('weight')
        plt.ylabel('target')
        plt.show()

        # Pairplot to see distributions and relationships
        sns.pairplot(df)
        plt.show()

        # Correlation matrix
        correlation_matrix = df.corr()
        print(correlation_matrix)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

        sns.pairplot(self.ih.metadf[['year', 'gender', 'canon']])
        for i in self.ih.metadf.columns:
            print(i)


    def regressor_pipeline(self):
        df, X, y = self.load_data()

        regressors = {
            'Linear Regression': LinearRegression(n_jobs=-1),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'Bagging': BaggingRegressor(n_jobs=-1),
            'Extra Trees': ExtraTreesRegressor(n_jobs=-1),
            'Support Vector Regression': SVR(),
            'XGBoost': XGBRegressor(eval_metric='logloss', n_jobs=-1),
            'K-Nearest Neighbors': KNeighborsRegressor(n_jobs=-1) 
        }

        pipelines = {}
        for name, regressor in regressors.items():
            pipelines[name] = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('regressor', regressor)
            ])

        metrics = ['MAE', 'MSE', 'RMSE', 'R2']
        results = {name: {metric: [] for metric in metrics} for name in pipelines}


        # Split data into training and test sets (15% for final testing)
        X_train_full, X_final_test, y_train_full, y_final_test = train_test_split(X, y, test_size=0.15, random_state=4)


        kf = KFold(n_splits=5, shuffle=True, random_state=4)
        for train_index, test_index in kf.split(X_train_full, y_train_full):

            X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
            y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]

            if self.test:
                # Small test sample
                sample_size = int(len(X_train) * 0.01)
                X_train = X_train.sample(n=sample_size, random_state=42)
                y_train = y_train.loc[X_train.index]

            for name, pipeline in pipelines.items():
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                results[name]['MAE'].append(mean_absolute_error(y_test, y_pred))
                results[name]['MSE'].append(mean_squared_error(y_test, y_pred))
                results[name]['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                results[name]['R2'].append(r2_score(y_test, y_pred))

                print(f'{name} - MAE: {mean_absolute_error(y_test, y_pred):.3f}, '
                    f'MSE: {mean_squared_error(y_test, y_pred):.3f}, '
                    f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}, '
                    f'R2: {r2_score(y_test, y_pred):.3f}')
                
        final_results = {name: {} for name in pipelines}
        for name, pipeline in pipelines.items():
            print(f'Retraining {name} on the full training data...')

            pipeline.fit(X_train_full, y_train_full)
            y_final_pred = pipeline.predict(X_final_test)

            final_results[name] = {
                'MAE': mean_absolute_error(y_final_test, y_final_pred),
                'MSE': mean_squared_error(y_final_test, y_final_pred),
                'RMSE': np.sqrt(mean_squared_error(y_final_test, y_final_pred)),
                'R2': r2_score(y_final_test, y_final_pred)
            }

            print(f'{name} on final test set - MAE: {mean_absolute_error(y_final_test, y_final_pred):.3f}, '
                f'MSE: {mean_squared_error(y_final_test, y_final_pred):.3f}, '
                f'RMSE: {np.sqrt(mean_squared_error(y_final_test, y_final_pred)):.3f}, '
                f'R2: {r2_score(y_final_test, y_final_pred):.3f}')

        print('Aggregating cross-validation results...')
        results_aggregated = {}
        for name, metrics in results.items():
            results_aggregated[name] = {metric: metrics[metric] for metric in metrics}
            for metric in metrics:
                results_aggregated[name][f'{metric} Mean'] = np.mean(metrics[metric])
                results_aggregated[name][f'{metric} Std'] = np.std(metrics[metric])

        results_df = pd.DataFrame(results_aggregated).T
        results_df.columns = [f'{metric} (Fold Metrics)' for metric in metrics] + [f'{metric} Mean' for metric in metrics] + [f'{metric} Std' for metric in metrics]
        results_df = results_df.round(3)

        print('Appending final test set results...')
        final_test_df = pd.DataFrame(final_results).T
        final_test_df.columns = [f'Final Test {metric}' for metric in metrics]

        results_combined_df = results_df.join(final_test_df)
        print('Final combined results:')
        print(results_combined_df)

        self.save_data(file_name=f'pipeline_results_{self.attr}.csv', data=results_combined_df)


