# ---------------------------
# Created by Yujia Yang   
# Created on 21.07.25
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
ensure_package('scipy')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


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
save_name = 'MPS.pdf'
save_path = data_dir / save_name

# read data
data_name = 'features.xlsx'
data_path = data_dir/data_name
df = pd.read_excel(data_path)


# calculate Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k, r) - 1)))


n_features = 13

# create Cramér's V similarity matrix
cramer_matrix = pd.DataFrame(np.zeros((n_features, n_features)),
                             columns=df.columns, index=df.columns)

for col1 in df.columns:
    for col2 in df.columns:
        if col1 == col2:
            cramer_matrix.loc[col1, col2] = 1.0  # set self-similarity to 1
        else:
            cramer_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

# draw heatmap as 2D MPS
plt.figure(figsize=(10, 8))
sns.heatmap(cramer_matrix, annot=True, cmap='viridis_r', vmin=0, vmax=1, square=True)
plt.tight_layout()

# save plot
plt.savefig(save_path)
plt.show()