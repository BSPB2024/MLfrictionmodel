# -*- coding: utf-8 -*-

#!pip install ydata-profiling

import pandas as pd
import numpy as np

df = pd.read_csv('all_data.csv')


df = df[df['Avg_SN'] >= 30]


df = df[df != -9999.000000].dropna()


# !pip install pycaret

# check version
from pycaret.utils import version



from pycaret.regression import *



from pycaret.regression import *
reg1 = setup(df, target='Avg_SN', session_id=42, log_experiment=False, experiment_name='tabular-playground-feb-2021', use_gpu=True)

compared_models = compare_models(fold=5, n_select=5)

#!pip install autoviz tune-sklearn ray[tune] scikit-optimize hyperopt optuna hpbandster ConfigSpace

#Create Model

#lightgbm = create_model('lightgbm')

lightgbm2 = create_model('lightgbm',
    bagging_fraction=0.7, bagging_freq=7, boosting_type='gbdt',
    class_weight=None, colsample_bytree=1.0, device='cpu',
    feature_fraction=0.5, importance_type='split',
    learning_rate=0.071, max_depth=-1, min_child_samples=70,
    min_child_weight=0.005, min_split_gain=0.9, n_estimators=250,
    n_jobs=-1, num_leaves=100, objective=None, random_state=42,
    reg_alpha=5, reg_lambda=0.15, silent=True, subsample=1.0,
    subsample_for_bin=200000, subsample_freq=0
)

#rf
rf= create_model('rf')

gb= create_model('gbr')

#tuning
lightgbm = tune_model(lightgbm2, n_iter=10, optimize='RMSE')
lightgbm

print(lightgbm2.dump_model())

#!pip install shap

best = automl(optimize='RMSE')

plot_model(best)
plot_model(lightgbm)

plot_model(best, plot='error')

plot_model(lightgbm, plot='feature')

evaluate_model(best)

# Train on all training data
final_model = finalize_model(lightgbm)


predict_model(lightgbm);

final_lightgbm = finalize_model(lightgbm)


print(lightgbm)

predict_model(lightgbm);

save_model(lightgbm,'Final Lightgbm Model  27 sep 2023')

data = df.sample(frac=0.9, random_state=786)
data_unseen = df.drop(data.index).reset_index(drop=True)
data.reset_index(drop=True, inplace=True)

print('Unseen Data For Predictions: ' + str(data_unseen.shape))

unseen_predictions = predict_model(lightgbm, data=data_unseen)
unseen_predictions.to_csv('predicted.csv',index = False)
unseen_predictions.head()

from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = [18, 6]

unseen_predictions.Avg_SN.plot(linewidth = 3, label = 'Actual', color = 'red')
unseen_predictions.prediction_label.plot(linewidth = 2, label = 'Predicted', color = 'blue', linestyle = '--')
plt.legend(fontsize = 'large')

#Saving the model
save_model(final_lightgbm,'Final Lightgbm Model 27 sep 2023')

print(final_lightgbm.dump_model())