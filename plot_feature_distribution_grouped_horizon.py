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
save_name = 'feature_distribution_features_grouped_horizontal.jpeg'
save_cohen = 'feature_distribution_cohend.csv'
save_path = data_dir / save_name
cohen_path = data_dir /save_cohen

# read data
data_name = 'statistic_info_lottery_pproblem_full_withpredict.csv'
data_path = data_dir/data_name
df = pd.read_csv(data_path)

# draw plot
colormap = plt.get_cmap('viridis')
colors = colormap(np.linspace(0.1, 1, 3))

fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax = ax.flatten()

# reorganize the data
features = ['p_risky', 'p_high_EV', 'p_high_mean']
labels = ['Risk taking', 'Expected value', 'Experienced mean']

df1 = df[df['incentivization'] == 'yes']
df1['categories'] = '1'
df1['group'] = 'Incentivization'
df2 = df[df['incentivization'] == 'no']
df2['categories'] = '2'
df2['group'] = 'Incentivization'
df3 = df[df['domain'] == 'gain']
df3['categories'] = '1'
df3['group'] = 'Domain'
df4 = df[df['domain'] == 'loss']
df4['categories'] = '2'
df4['group'] = 'Domain'
df5 = df[df['domain'] == 'mixed']
df5['categories'] = '3'
df5['group'] = 'Domain'
df6 = df[df['feedback_format'] == 'partial']
df6['categories'] = '1'
df6['group'] = 'Feedback format'
df7 = df[df['feedback_format'] == 'full']
df7['categories'] = '2'
df7['group'] = 'Feedback format'
df8 = df[df['problem_type'] == 'risky_risky']
df8['categories'] = '1'
df8['group'] = 'Problem type'
df9 = df[df['problem_type'] == 'risky_safe']
df9['categories'] = '2'
df9['group'] = 'Problem type'
df10 = df[df['identical_outcome'] == 1]
df10['categories'] = '1'
df10['group'] = 'Identical outcome'
df11 = df[df['identical_outcome'] == 0]
df11['categories'] = '2'
df11['group'] = 'Identical outcome'
df12 = df[df['free_samp'] == 1]
df12['categories'] = '1'
df12['group'] = 'Sampling'
df13 = df[df['free_samp'] == 0]
df13['categories'] = '2'
df13['group'] = 'Sampling'
df14 = df[df['feedback_type'] == 'outcome']
df14['categories'] = '1'
df14['group'] = 'Feedback type'
df15 = df[df['feedback_type'] == 'event']
df15['categories'] = '2'
df15['group'] = 'Feedback type'
df16 = df[df['stationarity'] == 0]
df16['categories'] = '2'
df16['group'] = 'Stationarity'
df17 = df[df['stationarity'] == 1]
df17['categories'] = '1'
df17['group'] = 'Stationarity'
df18 = df[df['n_trial'] == 1]
df18['categories'] = '1'
df18['group'] = 'Number of trials'
df19 = df[df['n_trial'] == 2]
df19['categories'] = '2'
df19['group'] = 'Number of trials'
df20 = df[df['n_trials'] == 3]
df20['categories'] = '3'
df20['group'] = 'Number of trials'


df_long =pd.concat([df6, df7, df14, df15, df16, df17, df10, df11, df12, df13, df1, df2, df8, df9, df3, df4, df5, df18, df19, df20])

df_long['group'] = df_long['group'].astype(str)

categories = ['Feedback format', 'Feedback type', 'Stationarity', 'Identical outcome', 'Sampling', 'Incentivization',  'Problem type',  'Domain', 'Number of trials']
category_sub = ['partial', 'full', '', 'outcome', 'event', '',  'stationary', 'dynamic', '', 'yes', 'no', '', 'yes', 'no', '', 'yes', 'no', '', 'risky-risky', 'risky-safe', '', 'gain', 'loss', 'mixed', '1-50', '51-100', '>100']


# add category labels to the plot
def add_label(axn, categories, category_sub, size):
    y_adjust = [0.265, 0, -0.265]
    index = 0
    positions = axn.get_yticks()
    for cat_idx, category in enumerate(categories):
        if cat_idx >= len(positions):
            continue
        y = positions[cat_idx]
        for i in y_adjust:
            if index < len(category_sub):
                axn.text(1.01, y - i, category_sub[index],
                         rotation=0, ha='left', va='center', fontsize=size, fontweight='bold',
                         transform=axn.get_yaxis_transform(), clip_on=False)
                index += 1


def compute_cohens_d(x1, x2):
    """compute Cohen's d"""
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    d = (np.mean(x1) - np.mean(x2)) / pooled_sd
    return d


# horizon stripplot
for i in range(len(features)):
    feature = features[i]
    dfn = df_long[df_long[feature].notna() & (df_long[feature] != '')]
    dfn = dfn.reset_index(drop=True)

    # stripplot
    sns.stripplot(y='group', x=feature, hue='categories',
                  hue_order=['1', '2', '3'],
                  data=dfn, jitter=True,
                  palette=colors,
                  dodge=True, size=4,
                  edgecolor='white',
                  ax=ax[i],
                  orient='h',
                  alpha=0.2,
                  legend=False,
                  zorder=2)

    # add mean point
    # calculate every group × category mean
    grouped_means = dfn.groupby(['group', 'categories'])[feature].mean().reset_index()

    # get ytick labels and the position
    group_labels = ax[i].get_yticks()
    group_names = ax[i].get_yticklabels()

    # mapping：group name → ytick
    y_mapping = {label.get_text(): y for label, y in zip(group_names, group_labels)}

    # draw mean point
    for _, row in grouped_means.iterrows():
        group = row['group']
        cat = row['categories']
        mean_val = row[feature]

        # adjust y according to hue order
        offset = {'1': -0.25, '2': 0.0, '3': 0.25}
        y_pos = y_mapping[group] + offset[cat]

        # get the color of the category
        color_index = int(cat) - 1
        color = colors[color_index]

        ax[i].scatter(
            mean_val, y_pos,
            s=440,
            facecolors=color,
            edgecolors='white',
            linewidths=4.5,
            zorder=3
        )

    ax[i].set_xlim(0, 1)
    ax[i].axvline(x=0.5, color='grey', linestyle='--', linewidth=1)

    ax[i].set_title(labels[i], fontsize=30, fontweight='bold')

    # add category label
    if i == 2:
        add_label(ax[i], categories, category_sub, 20)
    if i == 0:
        ax[i].tick_params(axis='y', labelsize=25)
    else:
        ax[i].set_yticks([])

    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

    if i != 5:
        ax[i].set_xlabel("Choice proportion", fontsize=28)
        ax[i].tick_params(axis='x', labelsize=20)
    else:
        ax[i].set_xlabel("", fontsize=20)
        ax[i].set_xticks([])

    ax[i].set_ylabel('', fontsize=25, fontweight='bold')

# adjust the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(False)
plt.tight_layout()
plt.savefig(save_path)
plt.show()


# calculate cohen's d
category_label_map = {
    'Feedback format': {'1': 'partial', '2': 'full'},
    'Feedback type': {'1': 'outcome', '2': 'event'},
    'Identical outcome': {'1': 'yes', '2': 'no'},
    'Stationarity': {'1': 'stationary', '2': 'dynamic'},
    'Sampling': {'1': 'yes', '2': 'no'},
    'Incentivization': {'1': 'yes', '2': 'no'},
    'Problem type': {'1': 'risky_risky', '2': 'risky_safe'},
    'Domain': {'1': 'gain', '2': 'loss', '3': 'mixed'},
    'Number of trials': {'1': '1-50', '2': '51-100', '3': '>100'}
}

cohen_results = []

groups = df_long['group'].unique()
features = ['p_high_EV', 'p_high_mean', 'p_risky']

for group in groups:
    for feature in features:
        sub_df = df_long[(df_long['group'] == group) & (df_long[feature].notna())]
        sub_df[feature] = sub_df[feature].astype(float)

        categories = sub_df['categories'].unique()
        if len(categories) < 2:
            continue

        for cat1, cat2 in combinations(categories, 2):
            data1 = sub_df[sub_df['categories'] == cat1][feature]
            data2 = sub_df[sub_df['categories'] == cat2][feature]

            if len(data1) > 1 and len(data2) > 1:
                try:
                    d = compute_cohens_d(data1, data2)
                    label1 = category_label_map.get(group, {}).get(cat1, cat1)
                    label2 = category_label_map.get(group, {}).get(cat2, cat2)
                    cohen_results.append({
                        'group': group,
                        'feature': feature,
                        'category1': cat1,
                        'category2': cat2,
                        'label1': label1,
                        'label2': label2,
                        'n1': len(data1),
                        'n2': len(data2),
                        'mean1': np.mean(data1),
                        'mean2': np.mean(data2),
                        'SD1': np.std(data1, ddof=1),
                        'SD2': np.std(data2, ddof=1),
                        'Cohen_d': d
                    })
                except Exception as e:
                    print(f"Error calculating d for {group}, {feature}, {cat1} vs {cat2}: {e}")


cohen_df = pd.DataFrame(cohen_results)
cohen_df.to_csv(cohen_path, index=False)
