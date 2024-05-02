import os
import random
# import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
                if assets_dir:
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
            print(f"Соотношение уникальных значений и общего количества записей в столбце '{col}': {dataset[col].nunique() / dataset.shape[0]:.2f}")

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

    def data_preprocessing(self):
        pass

    def model_fitting(self, model_name=None, features=None, labels=None, params=None):
        pass

    def model_logging(self,
					  experiment_name=None,
					  run_name=None,
					  registry_model=None,
					  params=None,
					  metrics=None,
					  model=None,
					  train_data=None,
					  test_data=None,
					  test_label=None,
					  metadata=None,
					  code_paths=None,
					  tsh=None,
					  tsp=None):

        mlflow.set_tracking_uri(f"http://{tsh}:{tsp}")
        mlflow.set_registry_uri(f"http://{tsh}:{tsp}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id
		
        if run_name == 'baseline_1_all_data':
            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                mlflow.log_metrics(metrics)
        else:
            pip_requirements = "requirements.txt"
            try:
                signature = mlflow.models.infer_signature(test_data, test_label.values)
            except:
                signature = mlflow.models.infer_signature(test_data, test_label)
            input_example = (pd.DataFrame(train_data)).iloc[0].to_dict()

            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                mlflow.log_artifacts(self.ASSETS_DIR)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="models",
                    pip_requirements=pip_requirements,
                    signature=signature,
                    input_example=input_example,
                    metadata=metadata,
                    code_paths=code_paths,
                    registered_model_name=registry_model,
                    await_registration_for=60
				)

    def model_analysis(self, model):
        pass

    def genre_rec_sys(self, dataset=None, image_name=None, emb_array=None, num_recommendations=3):
        pass