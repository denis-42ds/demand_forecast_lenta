import os
import random
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import psycopg2 as psycopg
import matplotlib.pyplot as plt

from typing import List, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_percentage_error
)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

sns.set_style('white')
sns.set_theme(style='whitegrid')

pd.options.display.max_rows = 32
pd.options.display.max_columns = 50

class DatasetExplorer:
    def __init__(self, DATA_PATH: str = None):
        self.DATA_PATH = DATA_PATH

    @staticmethod
    def wape(y_true: np.array, y_pred: np.array):
        # функция для расчёта метрики wape
        return np.sum(np.abs(y_true-y_pred))/np.sum(np.abs(y_true))

    def explore_dataset(self, target: str = None, assets_dir: str = None):

        dataset = pd.read_csv(self.DATA_PATH)
        print('Общая информация по набору данных:')
        dataset.info()
        print('\nПервые пять строк набора данных:')
        display(dataset.head(5))
        print('\nКоличество полных дубликатов строк:')
        display(dataset.duplicated().sum())
        if dataset.duplicated().sum() > 0:
            sizes = [dataset.duplicated().sum(), dataset.shape[0]]
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=['duplicate', 'not a duplicate'], autopct='%1.0f%%')
            plt.title('Number of complete duplicates in the total number of rows', size=12)
            if assets_dir:
                plt.savefig(os.path.join(assets_dir, 'Number of complete duplicates in\nthe total number of rows.png'))
            plt.show()

        print('\nКоличество пропущенных значений:')
        display(dataset.isnull().sum())
        if dataset.isnull().values.any():
            if dataset.shape[1] <= 20 or dataset.shape[0] < 1000000:
                sns.heatmap(dataset.isnull(), cmap=sns.color_palette(['#000099', '#ffff00']))
                plt.xticks(rotation=90)
                plt.title('Visualization of the number of missing values', size=12, y=1.02)
                if assets_dir is not None:
                    plt.savefig(os.path.join(assets_dir, 'Visualization of the number of missing values.png'))
                plt.show()

        print('\nПроцент пропущенных значений в признаках:')
        missing_values_ratios = {}
        for column in dataset.columns[dataset.isna().any()].tolist():
            missing_values_ratio = dataset[column].isna().sum() / dataset.shape[0]
            missing_values_ratios[column] = missing_values_ratio
        for column, ratio in missing_values_ratios.items():
            print(f"{column}: {ratio*100:.2f}%")

        # Исследование признаков, у которых в названии есть 'id'
        id_columns = [col for col in dataset.columns if 'id' in col]
        for col in id_columns:
            print(f"Количество уникальных значений в столбце '{col}': {dataset[col].nunique()}")
            print(f"Соотношение уникальных значений и общего количества записей в столбце '{col}': {dataset[col].nunique() / dataset.shape[0]:.4f}")

        if target:
            print('\nОписательные статистики целевой переменной:')
            display(dataset[target].describe())
            print()
            sns.set_palette("husl")
            sns.histplot(data=dataset, x=target, bins=10, log_scale=(True, False))
            plt.xlabel('Sales in units')
            plt.ylabel('Sales count')
            plt.title('Target total distribution')
            if assets_dir:
                plt.savefig(os.path.join(assets_dir, 'Target total distribution.png'))
            plt.show()

        return dataset

    @staticmethod
    def data_splitting(dataset: pd.DataFrame = None, pred_period: int = None, target: str = None):

        dataset.sort_index(inplace=True)
        y = dataset[target]
        X = dataset.drop([target], axis=1)
        X.index = pd.to_datetime(X.index)

        X_train = X[X.index <= (X.index.max() - pd.DateOffset(days=pred_period))]
        X_test = X[X.index > (X.index.max() - pd.DateOffset(days=pred_period))]
        y_train = y[:X_train.shape[0]]
        y_test = y[X_train.shape[0]:]

        print('Размерности полученных выборок:')
        display(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def feature_engineering(dataset: pd.DataFrame = None,
                            target: str = None,
                            date_column: str = None,
                            group_columns: List[str] = None,
                            window_size: int = 14) -> pd.DataFrame:

        dataset = dataset.sort_values(by=date_column)
        dataset.reset_index(drop=True, inplace=True)
        dataset['rolling_mean_14d'] = dataset.groupby(group_columns)[target].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        dataset['rolling_mean_14d_lag3'] = dataset['rolling_mean_14d'].shift(3)
        dataset['rolling_mean_14d_lag7'] = dataset['rolling_mean_14d'].shift(7)
        dataset['rolling_mean_14d_lag11'] = dataset['rolling_mean_14d'].shift(11)
        dataset['rolling_mean_14d_lag14'] = dataset['rolling_mean_14d'].shift(14)
        
        dataset['month'] = dataset[date_column].dt.month
        dataset['day'] = dataset[date_column].dt.day
        dataset['day_of_year'] = dataset[date_column].dt.dayofyear
        dataset['day_of_week'] = dataset[date_column].dt.dayofweek
        dataset['quarter'] = dataset[date_column].dt.quarter
        dataset['is_month_start'] = dataset[date_column].dt.is_month_start.astype(int)
        dataset['is_month_end'] = dataset[date_column].dt.is_month_end.astype(int)

        dataset = dataset.dropna()
        dataset.drop('rolling_mean_14d', axis=1, inplace=True)
        dataset.set_index('date', inplace=True)
        print('Верхние строки набора данных:')
        display(dataset.head())

        return dataset

    @staticmethod
    def model_fitting(model_name: str = None,
                      train_features: pd.DataFrame = None,
                      train_labels: pd.DataFrame = None,
                      assets_dir: str = None,
                      tscv: TimeSeriesSplit = None,
                      params: Dict = None,
                      params_selection: bool = False):

        binary_features = []#train_features.loc[:, train_features.nunique() == 2].columns.to_list()
        cat_features = train_features.select_dtypes(include=['object']).columns.to_list()
        num_features = train_features.drop(binary_features+cat_features, axis=1).columns.to_list()

        preprocessor = ColumnTransformer(
            [
                ('binary', OneHotEncoder(drop='if_binary'), binary_features),
                ('cat', CatBoostEncoder(random_state=RANDOM_STATE), cat_features),
                ('numeric', StandardScaler(), num_features)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )

        if model_name == 'Linear Regression':
            model = LinearRegression()
        elif model_name == 'Random Forest':
            model = RandomForestRegressor()
        elif model_name == 'LGBM':
            model = lgb.LGBMRegressor()

        pipeline = Pipeline(
            [
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )

        metrics = {
            'wape': [],
            'mape': [],
            'rmse': [],
            'mae': [],
            'mse': [],
            'r2': []
        }

        if params_selection:
            randomized_search = RandomizedSearchCV(pipeline,
                                                   param_distributions=params,
                                                   n_iter=10,
                                                   cv=tscv,
                                                   scoring=make_scorer(DatasetExplorer.wape, greater_is_better=False))
            randomized_search.fit(train_features, train_labels)
            best_params = randomized_search.best_params_

        else:
            best_params = params

        pipeline.set_params(**best_params)
        for train_index, test_index in tscv.split(train_features):
            X_train_fold, X_test_fold = train_features.iloc[train_index], train_features.iloc[test_index]
            y_train_fold, y_test_fold = train_labels.iloc[train_index], train_labels.iloc[test_index]

            pipeline.fit(X_train_fold, y_train_fold)
            y_pred = pipeline.predict(X_test_fold)

            metrics['wape'].append(round(DatasetExplorer.wape(y_test_fold, y_pred), 3))
            metrics['mape'].append(round(mean_absolute_percentage_error(y_test_fold, y_pred), 3))
            metrics['rmse'].append(round(root_mean_squared_error(y_test_fold, y_pred), 3))
            metrics['mae'].append(round(mean_absolute_error(y_test_fold, y_pred), 3))
            metrics['mse'].append(round(mean_squared_error(y_test_fold, y_pred), 3))
            metrics['r2'].append(round(r2_score(y_test_fold, y_pred), 3))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        ax1.plot(range(1, len(metrics['wape'])+1), metrics['wape'], label='WAPE')
        ax1.plot(range(1, len(metrics['mape'])+1), metrics['mape'], label='MAPE')
        ax1.plot(range(1, len(metrics['r2'])+1), metrics['r2'], label='R2')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Metric values during cross validation: WAPE, MAPE, R2')
        ax1.legend()

        ax2.plot(range(1, len(metrics['rmse'])+1), metrics['rmse'], label='RMSE')
        ax2.plot(range(1, len(metrics['mae'])+1), metrics['mae'], label='MAE')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Metric values during cross validation: RMSE, MAE')
        ax2.legend()

        ax3.plot(range(1, len(metrics['mse'])+1), metrics['mse'], label='MSE')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Metric values during cross validation: MSE')
        ax3.legend()

        plt.tight_layout()
        if assets_dir is not None:
            plt.savefig(os.path.join(assets_dir, f"Metric values during cross validation {model_name} model.png"))
        plt.show()
        
        valid_metrics = {metric: round(np.mean(values), 3) for metric, values in metrics.items()}
        print('Средние значения метрик по кросс-валидации:')
        display(valid_metrics)

        return valid_metrics, pipeline

    @staticmethod
    def model_logging(experiment_name: str = None,
					  run_name: str = None,
					  registry_model: str = None,
					  params: Dict = None,
					  metrics: Dict = None,
					  model: Any = None,
					  train_data: pd.DataFrame = None,
                      train_label: pd.DataFrame = None,
					  assets_dir: str = None,
					  metadata: Dict = None,
					  code_paths: str = None,
					  tsh: str = None,
					  tsp: str = None):

        mlflow.set_tracking_uri(f"http://{tsh}:{tsp}")
        mlflow.set_registry_uri(f"http://{tsh}:{tsp}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id
		
        pip_requirements = 'requirements.txt'
        try:
            signature = mlflow.models.infer_signature(train_data.values, train_label.values)
        except:
            signature = mlflow.models.infer_signature(train_data, train_label)
        # input_example = (pd.DataFrame(train_data)).iloc[0].to_dict()
        input_example = train_data[:10]

        if 'LGBM' in registry_model:
            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                mlflow.log_artifacts(assets_dir)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                model_info = mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path='models',
                    pip_requirements=pip_requirements,
                    signature=signature,
                    input_example=input_example,
                    metadata=metadata,
                    code_paths=code_paths,
                    registered_model_name=registry_model,
                    await_registration_for=60
				)
        else:
            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                mlflow.log_artifacts(assets_dir)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path='models',
                    pip_requirements=pip_requirements,
                    signature=signature,
                    input_example=input_example,
                    metadata=metadata,
                    code_paths=code_paths,
                    registered_model_name=registry_model,
                    await_registration_for=60
				)

    @staticmethod
    def models_comparison(connection: Dict = None,
                          postgres_credentials: Dict = None,
                          experiment_name: str = None,
                          assets_dir: str = None):
        connection.update(postgres_credentials)
        with psycopg.connect(**connection) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                            SELECT
                              experiments.name AS experiment_name,
                              runs.name AS run_name,
                              model_versions.name AS model_name,
                              model_versions.version AS model_version,
                              MAX(CASE WHEN metrics.key = 'wape' THEN metrics.value END) AS wape,
                              MAX(CASE WHEN metrics.key = 'mape' THEN metrics.value END) AS mape,
                              MAX(CASE WHEN metrics.key = 'rmse' THEN metrics.value END) AS rmse,
                              MAX(CASE WHEN metrics.key = 'mae' THEN metrics.value END) AS mae,
                              MAX(CASE WHEN metrics.key = 'mse' THEN metrics.value END) AS mse,
                              MAX(CASE WHEN metrics.key = 'r2' THEN metrics.value END) AS r2
                            FROM experiments
                              LEFT JOIN runs USING (experiment_id)
                              LEFT JOIN metrics USING (run_uuid)
                              LEFT JOIN model_versions ON model_versions.run_id=runs.run_uuid
                            WHERE
                              experiments.name = %s
                            GROUP BY
                              experiments.name,
                              runs.name,
                              model_versions.name,
                              model_versions.version
                            ORDER BY wape DESC
                            ''', (experiment_name,))
                table_data = cur.fetchall()
                table_columns = [desc[0] for desc in cur.description]
                print('Models and their metrics:')
                models_data = pd.DataFrame(table_data, columns=table_columns)
                display(models_data)

        plt.figure(figsize=(14, 8))
        sns.set_palette("husl")
        metrics = ['wape', 'mape', 'r2', 'mse', 'rmse', 'mae']

        for i, metric in enumerate(metrics):
            if i < 3:
                plt.subplot(2, 3, i+1)
            else:
                plt.subplot(2, 3, i+1)
            sns.barplot(x='model_name', y=metric, data=models_data, hue='model_name')
            plt.title(f'Comparison of {metric.upper()}')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        if assets_dir is not None:
            plt.savefig(os.path.join(assets_dir, 'Comparisons of models metrics.png'))
        plt.show()

    @staticmethod
    def test_best_model(model_name: str = None,
                        model: Any = None,
                        features_train: pd.DataFrame = None,
                        features_test: pd.DataFrame = None,
                        target_test: pd.DataFrame = None,
                        save_figure: bool = False):
        y_pred_proba = model.predict_proba(features_test.values)[:, 1]
        roc_auc_value = roc_auc_score(target_test, y_pred_proba)
        y_pred = model.predict(features_test.values)
        f1_value = f1_score(target_test, y_pred)
        
        print(f"ROC-AUC на тестовой выборке: {round(roc_auc_value, 2)}")
        print(f"F1 на тестовой выборке: {round(f1_value, 2)}")

        fig, axs = plt.subplots(1, 2)
        fig.tight_layout(pad=1.0)
        fig.set_size_inches(18, 6, forward=True)

        sns.heatmap(confusion_matrix(target_test, y_pred.round()), annot=True, fmt='3.0f', cmap='crest', ax=axs[0])
        axs[0].set_title('Test confusion matrix', fontsize=16, y=1.02)

        if model_name == 'LGBM':
            lgb.plot_importance(model,
                                ax=axs[1],
                                height=0.2,
                                xlim=None,
                                ylim=None,
                                title='Feature importance',
                                xlabel='Feature importance',
                                ylabel='Features',
                                importance_type='auto',
                                max_num_features=None,
                                ignore_zero=True,
                                figsize=None,
                                dpi=None,
                                grid=True,
                                precision=3)
        else:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer.shap_values(features_train)
            shap.summary_plot(shap_values, features_train, plot_size=(14, 5), show=False, plot_type='bar', ax=axs[1])
        if save_figure:
            plt.savefig(os.path.join(ASSETS_DIR, 'Test confusion matrix and Features importance.png'))
        plt.show()