# ---------------------------
# Created by Yujia Yang   
# Created on 21.12.25
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
ensure_package('seaborn')
ensure_package('matplotlib')
ensure_package('itertools')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')


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
save_name = 'feature_distribution_paradigm_horizontal_withpredict.pdf'
save_cohen = 'paradigm_distribution_cohen_d.csv'
save_path = data_dir / save_name
cohen_path = data_dir /save_cohen

# read data
data_name = 'statistic_info_lottery_pproblem_full_withpredict.csv'
data_path = data_dir/data_name
df = pd.read_csv(data_path)

# draw plot
colormap = plt.get_cmap('viridis')
colors = colormap(np.linspace(0.1, 1, 10))
fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax = ax.flatten()

features = ['p_risky', 'p_high_EV', 'p_high_mean']
predict_features = ['p_risky_pre', 'p_high_EV_pre', 'p_high_mean_pre']
labels = ['Risk Preference', 'Expected Value', 'Experienced Value']
paradigm_order = ['free sampling', 'regulated sampling', 'lottery bandits', 'description bandits', 'dynamic bandits', 'probability learning', 'binary prediction', 'social binary prediction', 'other']


def compute_cohens_d(x1, x2):
    """calculate Cohen's d"""
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    d = (np.mean(x1) - np.mean(x2)) / pooled_sd
    return d


# draw the stripplot
for i in range(0, len(features)):
    feature = features[i]
    predict_feature = predict_features[i]
    df1 = df[df[feature] != '']

    # add scatter
    sns.stripplot(y='paradigm', x=feature,
                  orient='h',
                  data=df1, jitter=True,
                  order=paradigm_order,
                  palette=colors,
                  size=5,
                  edgecolor='white',
                  ax=ax[i],
                  alpha=0.8,
                  legend=False,
                  zorder=2,
                 )

    # add mean point
    # calculate group mean
    grouped_means = df1.groupby(['paradigm'])[[feature, predict_feature]].mean().reset_index()

    # get ytick labels and the positions
    group_labels = ax[i].get_yticks()
    group_names = [label.get_text() for label in ax[i].get_yticklabels()]
    y_mapping = dict(zip(group_names, group_labels))
    # mapping the color
    color_mapping = dict(zip(paradigm_order, colors))

    # draw the mean points and model prediction points
    for _, row in grouped_means.iterrows():
        paradigm = row['paradigm']
        y_pos = y_mapping.get(paradigm, None)
        mean_val = row[feature]
        predict_mean_val = row[predict_feature]

        if y_pos is not None and not np.isnan(mean_val):
            color = color_mapping.get(paradigm, 'black')  # fallback in case not found
            ax[i].scatter(mean_val, y_pos, s=440,
                          facecolors=color, edgecolors='white',
                          linewidths=4.5, zorder=3)
            ax[i].scatter(predict_mean_val, y_pos, s=100,
                          facecolors='black', edgecolors='white',
                          linewidths=0.5, zorder=3)

    ax[i].set_xlim(0, 1)
    # Add reference line at 0.5
    ax[i].axvline(x=0.5, color='grey', linestyle='--', linewidth=1)

    # Set title and labels
    ax[i].set_title(labels[i], fontsize=30, fontweight='bold')

    if i != 5:
        ax[i].tick_params(axis='x', labelsize=20)
        ax[i].set_xlabel("Choice Proportion", fontsize=28)
    else:
        ax[i].set_xticks([])
        ax[i].set_xlabel('', fontsize=20, fontweight='bold')
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    if i == 0:
        ax[i].tick_params(axis='y', labelsize=25)
    else:
        ax[i].set_yticks([])
    ax[i].set_ylabel('', fontsize=20, fontweight='bold')

# remove spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# remove grid
plt.grid(False)

plt.tight_layout()
plt.savefig(save_path)
plt.show()

# calculate cohen's d
cohen_results = []

for feature in features:
    df1 = df[df[feature] != '']
    df1[feature] = df1[feature].astype(float)

    paradigms = df1['paradigm'].unique()

    for p1, p2 in combinations(paradigms, 2):
        data1 = df1[df1['paradigm'] == p1][feature]
        data2 = df1[df1['paradigm'] == p2][feature]

        if len(data1) > 1 and len(data2) > 1:
            try:
                d = compute_cohens_d(data1, data2)
                cohen_results.append({
                    'feature': feature,
                    'paradigm1': p1,
                    'paradigm2': p2,
                    'n1': len(data1),
                    'n2': len(data2),
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2),
                    'SD1': np.std(data1, ddof=1),
                    'SD2': np.std(data2, ddof=1),
                    'Cohen_d': d
                })
            except Exception as e:
                print(f"Error calculating d for {feature}, {p1} vs {p2}: {e}")

# save cohens'd table
cohen_df = pd.DataFrame(cohen_results)
cohen_df.to_csv(cohen_path, index=False)