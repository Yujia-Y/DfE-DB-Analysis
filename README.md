# About codes

## Introduction
Here we offer the analysis codes we use in our paper. I will briefly introduce each code below.

- **table_info_full**: we use this code to calculate the high EV choice proportion, high experienced mean choice proportion, and risk choice proportion. Also add all the feature info we needed for further analysis here. If you want to run this code, please change the data_path in line 59 to the path you save the database.

- **add_trial_num**: This code calculate the experienced trial number for each participant in each problem, and the mean value for all participants in each problem. The number include NA trials in the processed data. If you want to run this code, please change the data_path in line 54 to the path you save the database.

- **train_model_manualkfold**: In this code, we trained 3 models with RandomForest. Model1 with all paradigm, model2 with all features we mentioned in paper, model3 with the 4 key features we discovered. 

- **deviation_sqrt_twoway**: In this code, we calculate the deviation within paper, within problem, and within participants. 
- **deviation_bar_plot_descriptive**: This code is used to create figure3.

- **plot_paradigm_distribution**: This code is used to create figure4.

- **plot_feature_distribution_grouped_horizon**: This code is used to create figure5.

- **feature_MPS_plot**: This code is used to create figure S1