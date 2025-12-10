# ---------------------------
# Created by Yujia Yang   
# Created on 15.07.25
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
ensure_package('pathlib')


import pandas as pd
from pathlib import Path
import numpy as np

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
df1_path = data_dir/ 'statistic_info_lottery_pproblem_full.csv'
df2_path = data_dir/ 'statistic_info_lottery_pparticipant_full.csv'
# save path and name
df1_save = 'statistic_info_lottery_pproblem_full_add_trial.csv'
df2_save = 'statistic_info_lottery_pparticipant_full_add_trial.csv'
save_path_df1 = data_dir / df1_save
save_path_df2 = data_dir / df2_save

# set data path
base_dir = Path('Yourpath /DfE-DB/data/')  # input the path you save the database data

df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)

# problem key change A_B, B_A to AB one problem
df1["options2"] = df1["options"].str.split("_").apply(lambda x: "".join(sorted(x)))
df2["options2"] = df2["options"].str.split("_").apply(lambda x: "".join(sorted(x)))

df1['key'] = (
    df1['problem'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    + '_' +
    df1['condition'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    + '_' +
    df1['options2'].astype(str)
)

df2['key'] = (
    df2['problem'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    + '_' +
    df2['condition'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    + '_' +
    df2['options2'].astype(str)
)


trial_counts_all = pd.DataFrame()
trial_counts_problem = pd.DataFrame()
for paper in df1['paper'].unique():
    if paper == 'RNW2015':
        continue
    for study in df1['study'].unique():
        processed_path = base_dir / paper / 'processed'
        data_file = processed_path / f"{paper}_{study}_data.csv"

        if data_file.exists():
            df_data = pd.read_csv(data_file)
            df_data = df_data[(df_data['stage'].isna()) | (df_data['stage'] == 2)]
            df_data = df_data[(df_data['outcome'].str.contains(':', case=False, na=False)) | (df_data['outcome'] == '')].reset_index()

            # check all columns need exist
            if all(col in df_data.columns for col in ['subject', 'problem', 'condition']):
                if df_data['condition'].isna().any():
                    df_data['condition'] = df_data['condition'].fillna(12)
                df_data["options2"] = (
                    df_data["options"]
                    .apply(lambda x: "".join(sorted(str(x).split("_"))) if isinstance(x, str) and "_" in x else np.nan)
                )
                # turn to str
                df_data['problem_str'] = df_data['problem'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
                df_data['condition_str'] = df_data['condition'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
                df_data['options2_str'] = df_data['options2'].fillna('').astype(str)

                df_data['key'] = df_data[['problem_str', 'condition_str', 'options2_str']].agg('_'.join, axis=1)

                # trial number for each participant in each problem
                trial_counts = (
                    df_data.groupby(['subject', 'key'])
                    .size()
                    .rename("n_trials")
                    .reset_index()
                )
                # trials per problem
                mean_trials_per_problem = (
                    trial_counts.groupby("key")["n_trials"]
                    .mean()
                    .reset_index()
                )

                #subject trial
                trial_counts['paper'] = paper
                trial_counts['study'] = study
                trial_counts["subject"] = trial_counts["subject"].astype(str)
                trial_counts["key"] = trial_counts["key"].astype(str)
                trial_counts_all = pd.concat([trial_counts_all, trial_counts])
                # problem trial
                mean_trials_per_problem['paper'] = paper
                mean_trials_per_problem['study'] = study
                mean_trials_per_problem["key"] = mean_trials_per_problem["key"].astype(str)
                trial_counts_problem = pd.concat([trial_counts_problem, mean_trials_per_problem])
            else:
                print(f"no 'subject' or 'problem' column: {data_file}")
        else:
            print(f"data not found: {data_file}")

df1['key'] = df1['key'].astype(str).str.strip()
df1["paper"] = df1["paper"].astype(str)
df1["study"] = df1["study"].astype(str)
df2["paper"] = df2["paper"].astype(str)
df2["study"] = df2["study"].astype(str)
df2['key'] = df2['key'].astype(str).str.strip()


trial_counts_problem["paper"] = trial_counts_problem["paper"].astype(str)
trial_counts_problem["study"] = trial_counts_problem["study"].astype(str)
trial_counts_problem['key'] = trial_counts_problem['key'].astype(str).str.strip()

df1 = df1.merge(
    trial_counts_problem,
    left_on=['paper', 'study', "key"],
    right_on=['paper', 'study', "key"],
    how="left"
        )
df2 = df2.merge(
    trial_counts_all,
    left_on=['paper', 'study', "participant", "key"],
    right_on=['paper', 'study', "subject", "key"],
    how="left"
        )


df1.to_csv(save_path_df1, index=False)
df2.to_csv(save_path_df2)