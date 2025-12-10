# ---------------------------
# Created by Yujia Yang   
# Created on 04.03.25
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

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
save_name = 'eviation_'
save_path = data_dir / save_name

# read data
data_name = 'statistic_info_lottery_pparticipant_full.csv'
data_path = data_dir/data_name
data = pd.read_csv(data_path)

# set group key
data['study_name'] = data['paper'].astype(str) + '_' + data['study'].astype(str)
# Standardize options (ensure A_B and B_A are the same)
data['standard_options'] = data['options'].apply(lambda x: '_'.join(sorted(x.split('_'))))
data['problem_name'] = data['paper'].astype(str) + '_' + data['study'].astype(str) + '_' + data['problem'].astype(str) + '_' + data['condition'].astype(str) + '_' + data['standard_options'].astype(str)
data['participant_name'] = data['paper'].astype(str) + '_' + data['study'].astype(str) + '_' + data['participant'].astype(str)
# Compute overall means
overall_means = data[['p_high_EV', 'p_high_mean', 'p_risky']].mean(skipna=True)

# road 1, paper, study, problem, participant
study_results, problem_results = [], []

# road 2, paper, study, participant, problem
participant_results = []

# group by paper
for study, data_study in data.groupby('study_name'):
    print(f'Processing: {study}')
    # Compute study-level deviation
    study_means = data_study[['p_high_EV', 'p_high_mean', 'p_risky']].mean(skipna=True)
    study_deviation = (study_means - overall_means) ** 2 * len(data_study)
    study_results.append(pd.Series({'study': study, **study_deviation}))

    # Group by standardized options
    for problem, data_problem in data_study.groupby('problem_name'):
        problem_means = data_problem[['p_high_EV', 'p_high_mean', 'p_risky']].mean(skipna=True)
        problem_deviation = (problem_means - overall_means) ** 2 * len(data_problem)
        problem_results.append(pd.Series(
            {'study': study, 'problem': data_problem['problem'].iloc[0], 'options': data_problem['options'].iloc[0],
             **problem_deviation}))

    # group by participants
    for participant, data_participant in data_study.groupby('participant_name'):
        participant_means = data_participant[['p_high_EV', 'p_high_mean', 'p_risky']].mean(skipna=True)
        participant_deviation = (participant_means - overall_means) ** 2 * len(data_participant)
        participant_results.append(pd.Series(
            {'study': study, 'participant': data_participant['participant'].iloc[0],
             **participant_deviation}))

# Convert lists to DataFrames
df_study = pd.DataFrame(study_results)
df_problem = pd.DataFrame(problem_results)
df_participant = pd.DataFrame(participant_results)


# Compute sqrt values for participant, problem, study, and paper level deviations for both road.
data_sqrt = pd.DataFrame({
    'ev_participant': np.sqrt(df_participant['p_high_EV'].sum(skipna=True)/(df_participant['p_high_EV'].count() - 1)),
    'ev_problem': np.sqrt(df_problem['p_high_EV'].sum(skipna=True)/(df_problem['p_high_EV'].count() - 1)),
    'ev_study': np.sqrt(df_study['p_high_EV'].sum(skipna=True)/(df_study['p_high_EV'].count() - 1)),

    'mean_participant': np.sqrt(df_participant['p_high_mean'].sum(skipna=True)/(df_participant['p_high_mean'].count() - 1)),
    'mean_problem': np.sqrt(df_problem['p_high_mean'].sum(skipna=True)/(df_problem['p_high_mean'].count() - 1)),
    'mean_study': np.sqrt(df_study['p_high_mean'].sum(skipna=True)/(df_study['p_high_mean'].count() - 1)),

    'risky_participant': np.sqrt(df_participant['p_risky'].sum(skipna=True)/(df_participant['p_risky'].count() - 1)),
    'risky_problem': np.sqrt(df_problem['p_risky'].sum(skipna=True)/(df_problem['p_risky'].count() - 1)),
    'risky_study': np.sqrt(df_study['p_risky'].sum(skipna=True)/(df_study['p_risky'].count() - 1)),


}, index=[0])  # Convert to DataFrame with one row


# Save results
df_study.to_csv(save_path + 'study.csv', index=False)
df_problem.to_csv(save_path + 'problem_way1.csv', index=False)
df_participant.to_csv(save_path + 'participant_way2.csv', index=False)
data_sqrt.to_csv(save_path + 'sqrt2.csv', index=False)

print("Computation finished!")
