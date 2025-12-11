# ---------------------------
# Created by Yujia Yang   
# Created on 19.06.25
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
ensure_package('matplotlib')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# save path and name
save_name = 'descriptive_bar_plot.jpeg'
save_path = data_dir / save_name

# read data
data_name = 'statistic_info_lottery_pparticipant_full.csv'
data_path = data_dir/data_name
data = pd.read_csv(data_path)

# create label
data["options2"] = data["options"].str.split("_").apply(lambda x: "".join(sorted(x)))

data['Study'] = data['paper'].astype(str) + "_" + data['study'].astype(str)
data['Problem'] = data['paper'].astype(str) + "_" + data['study'].astype(str) + "_" + data['condition'].astype(str) + "_" + data['problem'].astype(str) + "_" + data['options2'].astype(str)
data['Participant'] = data['paper'].astype(str) + "_" + data['study'].astype(str) + "_" + data['participant'].astype(str)

# set y and group labels
value_cols = ['p_risky', 'p_high_EV', 'p_high_mean']
value_titles = ['Risk taking', 'Expected Value', 'Experienced mean']
group_labels = ['Participant', 'Problem', 'Study']

# draw plot
fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

viridis_all = mpl.colormaps['viridis']
viridis = viridis_all(np.linspace(0, 0.8, 3))

for idx, (val_col, val_title) in enumerate(zip(value_cols, value_titles)):
    ax = axes[idx]
    df = data.dropna(subset=[val_col]).copy()
    means_per_group = []

    # calculate the mean value for each level
    for i, label in enumerate(group_labels):
        means = df.groupby(label)[val_col].mean().reset_index()
        means['group_type'] = label
        means['x'] = i
        means_per_group.append(means)

    plot_df = pd.concat(means_per_group, ignore_index=True)

    # draw scatter and mean value points
    for i, label in enumerate(group_labels):
        sub = plot_df[plot_df['group_type'] == label]
        y_raw = sub[val_col].values
        # add vertical jitter
        jitter_y = np.random.normal(loc=0, scale=0.01, size=len(sub))
        y = y_raw + jitter_y

        # add horizontal jitter
        x = np.random.normal(loc=i, scale=0.1, size=len(sub))
        color = viridis[i]

        # scatter
        alpha = 0.08 if i < 1 else 0.4
        ax.scatter(x, y, alpha=alpha, color=color, label=label, s=8)

        # mean value points
        group_mean = y.mean()
        edge_color = 'white' if i < 2 else 'black'
        ax.scatter(i, group_mean, color=color, s=80, edgecolor=edge_color, linewidth=1.2, zorder=10)


    # add title
    ax.set_title(val_title, fontsize=26, fontweight='bold')

    # add 0.5 line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2)

    # set up fpr x-axis
    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels(group_labels, rotation=0, fontsize=19)
    ax.set_xlabel('')

    # set up for y-axis, only remain for the left ax
    if idx == 0:
        ax.set_ylabel('Proportion', fontsize=24)
        ax.tick_params(axis='y', labelsize=19)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

# save fig
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(save_path)
plt.show()
