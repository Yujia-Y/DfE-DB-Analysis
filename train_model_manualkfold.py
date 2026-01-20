# ---------------------------
# Created by Yujia Yang   
# Created on 12.08.25
# ---------------------------
# install package first if not exist in current environment
import importlib
import subprocess
import sys
from pathlib import Path


def ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Package '{package_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


# import glob
ensure_package('pandas')
ensure_package('numpy')
ensure_package('sklearn')
ensure_package('shap')
ensure_package('joblib')


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
import joblib
import os


def get_names(name_pattern):
    current_dir = Path.cwd()  # get current working direction
    file_names = list(current_dir.glob(name_pattern))  # get matched file name list
    return file_names


# get current path
current_file = Path(__file__)

# get parent path
project_root = current_file.parent

# get data folder path
data_dir = project_root / 'data'

# save path and name
save_path = data_dir

# read data
data_name = 'statistic_info_lottery_pproblem_full_add_trial.csv'
data_path = data_dir/data_name
data = pd.read_csv(data_path)


# ========== Step 0: manualkfold ==========
def create_stratified_group_kfold(df, group_col='paper', stratify_col='key', n_splits=5, random_state=42):
    """like StratifiedGroupKFold：divided by problem to keep paper distribute equally between groups"""
    problem_df = df[[group_col, stratify_col]].drop_duplicates()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    problem_df['fold'] = -1
    for fold_idx, (_, test_idx) in enumerate(skf.split(problem_df, problem_df[stratify_col])):
        problem_df.loc[problem_df.index[test_idx], 'fold'] = fold_idx

    df = df.merge(problem_df[[group_col, 'fold']], on=group_col, how='left')
    return df


# ========== Step 1: load data ==========
df = pd.read_csv(data_path)
df['n_trial'] = np.select(
    [(df['n_trials'] <= 50), (df['n_trials'] <= 100)],
    [1, 2],
    default=3
)

# ========== Step 2: set up the model variables ==========
target_vars = ['p_risky', 'p_high_EV', 'p_high_mean']
model1_features = ['paradigm']
model2_features = ['feedback_format', 'feedback_type', 'identical_outcome', 'free_samp',
    'incentivization', 'problem_type', 'domain', 'stationarity', 'n_trial']
model3_features = ['feedback_type', 'identical_outcome', 'free_samp', 'stationarity']


# ========== Step 3: add k_fold ==========
df = create_stratified_group_kfold(df, group_col='problem', stratify_col='domain', n_splits=5)
df['key'] = df[model2_features].astype(str).agg('_'.join, axis=1)


# OneHot code（for model2 and model 3）
def build_preprocessor(cat_features):
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        drop=None  # keep all dummy columns
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop'
    )
    return preprocessor


# create save path
os.makedirs("models", exist_ok=True)

# record all results
results_list = []

# ========== Step 4: train the model ==========
for model_id, features in enumerate([model1_features, model2_features, model3_features], start=1):
    for target in target_vars:
        print(f"\n===== training model {model_id} - target variable: {target} =====")
        fold_results = []
        best_r2 = -np.inf
        best_model = None

        # add predict column
        pred_col = f"{target}_pre" if model_id == 2 else None

        for fold in range(5):
            train_df = df[df['fold'] != fold]
            test_df = df[df['fold'] == fold]
            train_df = train_df[~train_df[target].isna()]
            test_df = test_df[~test_df[target].isna()]

            X_train, y_train = train_df[features], train_df[target]
            X_test, y_test = test_df[features], test_df[target]

            preprocessor = build_preprocessor(features)
            model = RandomForestRegressor(random_state=42)

            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            fold_results.append(r2)
            print(f"Fold {fold} R²: {r2:.4f}")

            if model_id == 2:
                df.loc[test_df.index, pred_col] = y_pred

            if r2 > best_r2:
                best_r2 = r2
                best_model = pipe

                # for model 2 and 3, calculate Permutation Importance and SHAP
                if model_id in [2, 3]:
                    feature_names = X_test.columns.tolist()
                    result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)

                    # make sure the permutation importance and features number match
                    if len(result.importances_mean) != len(feature_names):
                        raise ValueError(
                            f"feature number {len(feature_names)} and importance number {len(result.importances_mean)} inconsistence")

                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance_mean': result.importances_mean,
                        'importance_std': result.importances_std
                    })
                    importance_df.to_csv(f"{save_path}models/model{model_id}_{target}_feature_importance.csv", index=False)

                    # SHAP
                    # get processed test data
                    X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
                    explainer = shap.TreeExplainer(best_model.named_steps['model'])
                    shap_values = explainer.shap_values(X_test_transformed)

                    # calculate the mean value for each feature
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

                    feature_names = best_model.named_steps['preprocessor'].named_transformers_[
                        'cat'].get_feature_names_out(features)

                    shap_df = pd.DataFrame({
                        'feature': feature_names,
                        'mean_abs_shap': mean_abs_shap
                    }).sort_values(by='mean_abs_shap', ascending=False)

                    # save to CSV
                    shap_df.to_csv(f"{save_path}models/model{model_id}_{target}_shap.csv", index=False)

        # save the r2
        mean_r2 = np.mean(fold_results)
        std_r2 = np.std(fold_results)
        for fold_idx, r2 in enumerate(fold_results):
            results_list.append({
                'model_id': model_id,
                'target': target,
                'fold': fold_idx,
                'r2': r2
            })
        results_list.append({
            'model_id': model_id,
            'target': target,
            'fold': 'mean',
            'r2': mean_r2
        })
        results_list.append({
            'model_id': model_id,
            'target': target,
            'fold': 'std',
            'r2': std_r2
        })

# save result to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(f"{save_path}models/model_performance_summary.csv", index=False)

# save df with prediction
df.to_csv(f"{save_path}statistic_info_lottery_pproblem_full_withpredict.csv", index=False)
