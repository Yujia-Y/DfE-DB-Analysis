# ---------------------------
# Created by Yujia Yang   
# Created on 26.05.2025
# change the rare event definition
# change the high mean calculation, count until both option has been chosen at least once, equal em leads to no hem.
# use problem + condition to divide problem
# add sort value by trial to the df when calculating experienced mean
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
ensure_package('os')


import pandas as pd
import numpy as np
import os
import glob
import warnings

# ignore RuntimeWarning
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
save_path = data_dir

# set data path
data_path = 'Yourpath /DfE-DB/data/'  # input the path you save the database data
paradigm_info_path = data_path / 'feature_table. xlsx'
os.chdir(data_path)
# create table
file_names = [save_path / 'statistic_info_lottery_pparticipant_full.csv',
              save_path / 'statistic_info_lottery_pproblem_full.csv']
columns_list = [['paper', 'study', 'condition', 'problem', 'participant', 'options', 'key', 'p_high_EV',
                 'p_high_mean', 'risky_option', 'p_risky', 'problem_type', 'paradigm', 'feedback_format',
                 'domain', 'incentivization', 'design', 'feedback_type', 'identical_outcome', 'free_samp', 'outcome_num', 'stationarity', 'num_feedback'],
                ['paper', 'study', 'condition', 'problem', 'n_participant', 'options', 'key', 'p_high_EV',
                 'p_high_mean', 'risky_option', 'p_risky', 'problem_type', 'paradigm', 'feedback_format',
                 'domain', 'incentivization', 'design', 'feedback_type', 'identical_outcome', 'free_samp', 'outcome_num', 'stationarity', 'num_feedback']]
for file, columns in zip(file_names, columns_list):
    empty_df = pd.DataFrame(columns=columns)
    empty_df.to_csv(file, index=False)

file_list = os.listdir(data_path)
# exclude specific folders, RNW2015 for have both partial and full feedback for one problem, YB2006 for condition 2, one participant has a full feedback every two trial
exclude_folders = ['.DS_Store', 'README.md', 'RNW2015', 'YB2006', 'feature_table.csv']
file_list = [folder for folder in file_list if folder not in exclude_folders]
file_list = sorted(file_list)

# get feature info
paradigm_info = pd.read_excel(paradigm_info_path)
paradigms = paradigm_info['paradigm'].unique()
paradigm_info['n_outcomes_per_option'] = (
    paradigm_info['n_outcomes_per_option']
    .replace({'2': 0, 'continuous': 2})
    .apply(lambda x: 1 if x not in [0, 2] else x)
).astype(int)
paradigm_info['stationarity'] = (
    paradigm_info['stationarity']
    .replace({'stationary': 0, 'dynamic': 1})
    .apply(lambda x: 2 if x not in [0, 1] else x)
).astype(int)
col = paradigm_info['numerical_feedback']

paradigm_info['numerical_feedback1'] = np.where(
    col.isin(['both', 'numerical;food', 'food;numerical']), 1,
    np.where(col == 'numerical', 0, 2)
).astype(int)


part_num = 20
# create processed df
df_processed_participant = pd.DataFrame(columns=columns_list[0])
df_processed_problem = pd.DataFrame(columns=columns_list[1])


# set functions
# calculate and record higher experienced mean option
def hem(df):
    high_mean_choice_record = [np.nan]  # hit or not
    options = df['choice'].unique()
    #  partial or full feedback
    df_full = df[df['outcome'].str.contains('_', case=False)]
    df_partial = df[~ df['outcome'].str.contains('_', case=False)]
    if not df_full.empty:  # full feedback
        # split data
        split_list = ['outcome1', 'outcome2', 'outcome3', 'outcome4', 'outcome5']
        max_outcomes = df_full['outcome'].str.count('_').max()
        df_full[split_list[0:max_outcomes + 1]] = df_full['outcome'].str.split('_', expand=True).apply(pd.Series)
        choice_list = []
        reward_list = []
        for out in split_list[0:max_outcomes + 1]:
            ch = 'choice' + out[-1:]
            re = 'reward' + out[-1:]
            choice_list.append(ch)
            reward_list.append(re)
            df_full[[ch, re]] = df_full[out].str.split(':', expand=True).apply(pd.Series)
        df_full = df_full.sort_values('trial').reset_index(drop=True)
        for i in range(1, len(df_full)):  # trial num
            data_experienced = df_full.iloc[0:i].reset_index(drop=True)
            df_merge = pd.DataFrame(columns=['choice', 'reward'])
            for j in range(0, len(choice_list)):
                df_merge = pd.concat([df_merge[['choice', 'reward']],
                                      data_experienced[[choice_list[j], reward_list[j]]].rename(
                                          columns={choice_list[j]: 'choice', reward_list[j]: 'reward'})])
            df_merge = df_merge.reset_index()
            df_merge['reward'] = df_merge['reward'].astype(float)
            reward_mean = df_merge.groupby('choice')['reward'].mean()
            reward_mean = reward_mean.astype(float)
            max_reward_mean = reward_mean.idxmax()  # return to index of the max mean, that is, the option with the highest experienced mean
            if df_full['choice'][i] == max_reward_mean:
                high_mean_choice_record.append(1)
            else:
                high_mean_choice_record.append(0)
    if not df_partial.empty:  # partial feedback
        df_partial[['choice_copy', 'reward']] = df_partial['outcome'].str.split(':', n=1).apply(
            pd.Series)  # split reward
        df_partial = df_partial.sort_values('trial').reset_index(drop=True)
        for i in range(1, len(df_partial)):
            data_experienced = df_partial.iloc[0:i].reset_index(drop=True)
            if len(data_experienced[
                       'choice_copy'].unique()) < 2:  # only high_experienced mean after experienced both options at least once
                high_mean_choice_record.append(np.nan)
            else:
                data_experienced['reward'] = data_experienced['reward'].astype(float)
                reward_mean = data_experienced.groupby('choice')['reward'].mean()
                reward_mean = reward_mean.astype(float)
                if reward_mean.nunique() == 1:
                    max_reward_mean = np.nan
                    high_mean_choice_record.append(np.nan)
                else:
                    max_reward_mean = reward_mean.idxmax()
                    if df_partial['choice'][i] == max_reward_mean:
                        high_mean_choice_record.append(1)
                    else:
                        high_mean_choice_record.append(0)

    p_high_mean = np.nanmean(high_mean_choice_record)
    return p_high_mean, high_mean_choice_record


# calculate value & prob
def info(option_info):
    outcome_values = option_info.filter(regex='^out').T.reset_index(drop=True).dropna()
    outcome_values.columns = range(len(outcome_values.columns))
    outcome_values = outcome_values[0].to_list()
    outcome_values = [float(num) for num in outcome_values]
    outcome_proportions = option_info.filter(regex='^pr').T.reset_index(drop=True).dropna()
    outcome_proportions.columns = range(len(outcome_proportions.columns))
    outcome_proportions = outcome_proportions[0].to_list()
    outcome_proportions = [float(num) for num in outcome_proportions]
    return outcome_values, outcome_proportions


# constant outcome
def identical_outcome(outcome_A, outcome_B):
    #     print(set(outcome_A))
    #     print(set(outcome_B))
    if set(outcome_A) == set(outcome_B):
        con_out = 1
    else:
        con_out = 0
    return con_out


# main
accumulated_paper = 0

for folder in file_list:  # paper
    target_path = os.path.join(data_path, folder)
    processed_path = os.path.join(target_path, 'processed')
    os.chdir(processed_path)
    data_name_list = get_names('*_data.csv')
    paper = folder
    print('paper' + str(accumulated_paper + 1))
    accumulated_paper += 1
    # read studies
    for data_name in data_name_list:  # study
        study = data_name[-10:-9]
        print(paper + '_' + study)
        basic_info = paradigm_info[paradigm_info['study'].str.startswith(paper + '_' + study)].reset_index()
        if basic_info['paradigm'][0] == 'observe or bet':
            #  have observed information but did not record in data
            print('observe or bet')
            continue
        df_study = pd.read_csv(data_name)
        # clearn data
        df_study = df_study[df_study['choice'] != '']
        df_study = df_study[df_study['outcome'] != '']
        df_study = df_study.dropna(subset=['choice', 'outcome', 'problem'])
        df_study = df_study[~pd.to_numeric(df_study['outcome'], errors='coerce').notna()]
        df_study = df_study[df_study['outcome'].str.contains(':', case=False,
                                                             na=False)].reset_index()  # filter only choice or only feedback data
        option_file = data_name[0:-9] + '_options.csv'
        option_inf = pd.read_csv(option_file)
        option_columns = option_inf.columns.tolist()
        if len(option_columns) != 5:  # choose studies with 2 outcomes per option
            print('no 2 out')
            continue
        values = option_inf[pd.to_numeric(option_inf.iloc[:, 1], errors='coerce').notna()]
        proportion = option_inf[pd.to_numeric(option_inf.iloc[:, 2], errors='coerce').notna()]

        if len(values) < len(option_inf) or len(proportion) < len(option_inf):
            print('no num')
            continue  # skip study with no numerical outcome or proportion
        # separate by condition
        if df_study['condition'].isna().any():
            df_study['condition'] = df_study['condition'].fillna(12)
        df_study["options2"] = df_study["options"].str.split("_").apply(lambda x: "".join(sorted(x)))

        df_study['key'] = df_study['problem'].astype(str) + '_' + df_study['condition'].astype(str) + '_' + df_study['options2']
        # problem = 1
        for key, df_problem in df_study.groupby('key'):
            df_problem = df_problem.reset_index(drop=True)
            condition = df_problem['condition'][0]
            problem = df_problem['problem'][0]
            df_s = pd.DataFrame(columns=df_processed_participant.columns)  # study level proportion data
            df_q = pd.DataFrame(columns=df_processed_problem.columns)  # problem level proportion data
            options = (
                df_problem["options"]
                .dropna()
                .str.split("_")
                .explode()  # explode to one column
                .unique()
            )
            if len(options) != 2:  # choose problem with 2 options
                problem += 1
                print('no 2 options')
                continue
            # create list to store subject level proportion
            subject_list = []
            p_HEV = []
            p_high_mean = []
            p_risky = []
            option_types = pd.DataFrame()
            i = 0  # count for number of options, to loc row
            high_ev_set = []
            var_set = []
            for k in range(0, len(options)):
                option = options[k]
                option_types.loc[i, 'option'] = option
                option_info = option_inf[option_inf['option'] == option].dropna(axis=1, how='all')
                if option_info.shape[1] == 3:
                    option_types.loc[i, 'type'] = 'safe'
                elif option_info.shape[1] == 5:
                    option_types.loc[i, 'type'] = 'risky'
                outcome_values, outcome_proportions = info(option_info)
                min_index = outcome_proportions.index(min(outcome_proportions))
                option_distribute = pd.concat([pd.Series(outcome_values), pd.Series(outcome_proportions)], axis=1)
                option_distribute.columns = ['value', 'proportion']
                option_distribute = option_distribute.apply(pd.to_numeric, errors='coerce')
                option_distribute = option_distribute.sort_values(by='value', ascending=True).reset_index(drop=True)
                # high ev choices
                ev = sum([x * y for x, y in zip(option_distribute['value'], option_distribute['proportion'])])
                var = sum(
                    [(x - ev) ** 2 * y for x, y in zip(option_distribute['value'], option_distribute['proportion'])])
                high_ev_set.append(ev)
                var_set.append(var)
                if k == 0 or ev > high_ev_option_value_p:
                    high_ev_option_value_p = ev
                    high_ev_option_p = option
                # high risk choices (high variance)
                if k == 0 or var > high_var_option_value_p:
                    high_var_option_value_p = var
                    risky_option_p = option
                # domain
                option_types.loc[i, 'min_proportion'] = min(outcome_proportions)
                min_index = outcome_proportions.index(min(outcome_proportions))
                option_types.loc[i, 'min_proportion_value'] = outcome_values[min_index]
                max_value_index = outcome_values.index(max(outcome_values))
                option_types.loc[i, 'max_value'] = max(outcome_values)
                option_types.loc[i, 'max_value_proportion'] = outcome_proportions[max_value_index]
                min_value_index = outcome_values.index(min(outcome_values))
                option_types.loc[i, 'min_value'] = min(outcome_values)
                option_types.loc[i, 'min_value_proportion'] = outcome_proportions[min_value_index]
                if min(outcome_values) >= 0:
                    option_types.loc[i, 'domain'] = 'gain'
                elif max(outcome_values) <= 0:
                    option_types.loc[i, 'domain'] = 'loss'
                else:
                    option_types.loc[i, 'domain'] = 'mixed'
                i += 1
            # hev
            if high_ev_set.count(high_ev_option_value_p) > 1:
                high_ev_option_p = 0
            # high var, if all options has same variance, then, no risky option
            if var_set.count(high_var_option_value_p) > 1:
                risky_option_p = 0
            # problem type
            problem_type = '_'.join(sorted(option_types['type'].astype(str)))
            if problem_type == 'safe_safe':  # skip only safe options problems
                problem += 1
                print('safe_safe')
                continue
            df_q.loc[0, 'problem_type'] = problem_type
            # define domain
            if option_types['domain'].eq('gain').all():
                df_q['domain'] = 'gain'
            elif option_types['domain'].eq('loss').all():
                df_q['domain'] = 'loss'
            else:
                df_q['domain'] = 'mixed'

            for subject in df_problem['subject'].unique():  # subject
                df_subject = df_problem[df_problem['subject'] == subject].reset_index(drop=True)
                if len(df_subject) <= 1:
                    continue
                subject_list.append(subject)
                # record higher expected value option
                if high_ev_option_p == 0:
                    p_hev_sub = np.nan
                else:
                    p_hev_sub = sum((df_subject['choice'] == high_ev_option_p).astype(int)) / len(df_subject)
                p_HEV.append(p_hev_sub)
                # calculate higher experienced mean option, trial 1 do not have data, record as 0
                p_high_mean_sub, high_mean_record = hem(df_subject)
                p_high_mean.append(p_high_mean_sub)
                # record risk choice
                if risky_option_p == 0:
                    p_risky_sub = np.nan
                else:
                    p_risky_sub = len(df_subject[df_subject['choice'] == risky_option_p]) / len(df_subject)
                p_risky.append(p_risky_sub)

            # define feedback format
            if '_' in df_problem['outcome'][0]:
                feedback_format = 'full'
            else:
                feedback_format = 'partial'

            # define constant outcome 1 = constant, 0 = not constant
            option_info_1 = option_inf[option_inf['option'] == options[0]].dropna(axis=1, how='all')
            outcome_value_1, outcome_proportions_1 = info(option_info_1)
            option_info_2 = option_inf[option_inf['option'] == options[1]].dropna(axis=1, how='all')
            outcome_value_2, outcome_proportions_2 = info(option_info_2)
            iden_out = identical_outcome(outcome_value_1, outcome_value_2)
            # record the subject level data
            df_s['participant'] = subject_list
            df_s['paper'] = paper
            df_s['study'] = study
            df_s['problem'] = problem
            df_s['condition'] = condition
            df_s['options'] = df_problem['options'][0]
            df_s['key'] = df_problem['key'][0]
            df_s['p_high_EV'] = p_HEV
            df_s['p_high_mean'] = p_high_mean
            df_s['risky_option'] = risky_option_p
            df_s['p_risky'] = p_risky
            df_s['problem_type'] = df_q['problem_type'][0]
            df_s['paradigm'] = basic_info['paradigm'][0]
            df_s['feedback_format'] = feedback_format
            df_s['domain'] = df_q['domain'][0]
            df_s['incentivization'] = basic_info['incentivization'][0]
            df_s['design'] = basic_info['design'][0]
            df_s['feedback_type'] = basic_info['feedback_type'][0]
            df_s['identical_outcome'] = iden_out
            df_s['free_samp'] = basic_info['sampling'][0]
            df_s['outcome_num'] = basic_info['n_outcomes_per_option'][0]
            df_s['stationarity'] = basic_info['stationarity'][0]
            df_s['num_feedback'] = basic_info['numerical_feedback1'][0]
            df_processed_participant = pd.concat([df_processed_participant, df_s])
            # calculate the problem level data
            df_q['paper'] = paper
            df_q['study'] = study
            df_q['problem'] = problem
            df_q['condition'] = condition
            df_q['n_participant'] = len(df_problem['subject'].unique())
            df_q['options'] = df_problem['options'][0]
            df_q['key'] = df_problem['key'][0]
            df_q['p_high_EV'] = np.nanmean(p_HEV)
            df_q['p_high_mean'] = np.nanmean(p_high_mean)
            df_q['risky_option'] = risky_option_p
            df_q['p_risky'] = np.nanmean(p_risky)
            df_q['paradigm'] = basic_info['paradigm'][0]
            df_q['feedback_format'] = feedback_format
            df_q['incentivization'] = basic_info['incentivization'][0]
            df_q['design'] = basic_info['design'][0]
            df_q['feedback_type'] = basic_info['feedback_type'][0]
            df_q['identical_outcome'] = iden_out
            df_q['free_samp'] = basic_info['sampling'][0]
            df_q['outcome_num'] = basic_info['n_outcomes_per_option'][0]
            df_q['stationarity'] = basic_info['stationarity'][0]
            df_q['num_feedback'] = basic_info['numerical_feedback1'][0]

            df_processed_problem = pd.concat([df_processed_problem, df_q])


# save data
df_processed_participant.to_csv(file_names[0], index=False)
df_processed_problem.to_csv(file_names[1], index=False)
