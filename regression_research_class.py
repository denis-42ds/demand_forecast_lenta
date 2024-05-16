import os
import random
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import psycopg2 as psycopg
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

sns.set_style("white")
sns.set_theme(style="whitegrid")

pd.options.display.max_rows = 32
pd.options.display.max_columns = 50

class DatasetExplorer:
    def __init__(self, DATA_PATH=None):
        self.DATA_PATH = DATA_PATH

    def explore_dataset(self, target=None, assets_dir=None):
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

    def data_splitting(self, dataset=None, pred_period=None, target=None):
        dataset.sort_index(inplace=True)
        y = dataset[target]
        X = dataset.drop([target], axis=1)

        X_train = X[X.index <= (X.index.max() - pd.DateOffset(days=pred_period))]
        X_test = X[X.index > (X.index.max() - pd.DateOffset(days=pred_period))]
        y_train = y[:X_train.shape[0]]
        y_test = y[X_train.shape[0]:]

        print('Размерности полученных выборок:')
        display(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        return X_train, y_train, X_test, y_test

    def data_preprocessing(self):
        pass

    def model_fitting(self, model_name=None, train_features=None, train_labels=None, tscv=None, params=None, params_selection=False):

        # функция для расчёта метрики
        def wape(y_true: np.array, y_pred: np.array):
            return np.sum(np.abs(y_true-y_pred))/np.sum(np.abs(y_true))

        if model_name == 'Baseline':
            model = LinearRegression()

        binary_features = train_features.loc[:, train_features.nunique() == 2].columns.to_list()
        cat_features = ['st_id', 'pr_group_id']

        preprocessor = ColumnTransformer(
            [
                ('binary', OneHotEncoder(drop='if_binary'), binary_features),
                ('cat', CatBoostEncoder(), cat_features)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )

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
        for train_index, test_index in tscv.split(train_features):
            X_train_fold, X_test_fold = train_features.iloc[train_index], train_features.iloc[test_index]
            y_train_fold, y_test_fold = train_labels.iloc[train_index], train_labels.iloc[test_index]
    
            pipeline.fit(X_train_fold.drop('pr_sku_id', axis=1), y_train_fold)
            y_pred = pipeline.predict(X_test_fold.drop('pr_sku_id', axis=1))
    
            metrics['wape'].append(round(wape(y_test_fold, y_pred), 3))
            metrics['mape'].append(round(mean_absolute_percentage_error(y_test_fold, y_pred), 3))
            metrics['rmse'].append(round(root_mean_squared_error(y_test_fold, y_pred), 3))
            metrics['mae'].append(round(mean_absolute_error(y_test_fold, y_pred), 3))
            metrics['mse'].append(round(mean_squared_error(y_test_fold, y_pred), 3))
            metrics['r2'].append(round(r2_score(y_test_fold, y_pred), 3))

        valid_metrics = {metric: round(np.mean(values), 3) for metric, values in metrics.items()}
        print('Средние значения метрик по кросс-валидации:')
        display(valid_metrics)

        return valid_metrics, pipeline

    def model_logging(self,
					  experiment_name=None,
					  run_name=None,
					  registry_model=None,
					  params=None,
					  metrics=None,
					  model=None,
					  train_data=None,
                      train_label=None,
					  assets_dir=None,
					  metadata=None,
					  code_paths=None,
					  tsh=None,
					  tsp=None):

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

        if 'catboost' in registry_model:
            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                mlflow.log_artifacts(assets_dir)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                model_info = mlflow.catboost.log_model(
                    cb_model=model,
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

    def models_comparison(self, connection=None, postgres_credentials=None, experiment_name=None, assets_dir=None):
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
                            ORDER BY wape
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
            sns.barplot(x='model_name', y=metric, data=models_data)
            plt.title(f'Comparison of {metric.upper()}')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        if assets_dir is not None:
            plt.savefig(os.path.join(assets_dir, 'Comparisons of models metrics.png'))
        plt.show()

    def genre_rec_sys(self, dataset=None, image_name=None, emb_array=None, num_recommendations=3):
        pass