

import os
import pandas as pd
import numpy as np
from models.random_forest import *
from models.decision_tree import *
from models.elastic_net_regression import *
from models.gradientBoostingRegressor import *
from models.lasso import *
from models.linear_regression import *
from models.ridge import *
from models.rnn import *
from models.svr import *
from models.xgboost import *
from utility.date_functions import *


# we will just use relative path, since the datasets are stored in the same directory as the models. 
directory = "datasets"

for filename in os.listdir(directory): 
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # freq = infer_frequency(df)
    freq = 'D'
    exog = create_time_features(df=df, freq=freq)
    
    lags = 7
    
    print(exog)
    
    results_ridge = forecast_and_evaluate_ridge(df_arg=df, exog=exog, lag_value=lags)
    results_rf = forecast_and_evaluate_random_forest(df_arg=df, exog=exog, lag_value=lags)
    results_lr = forecast_and_evaluate_linear_regression(df_arg=df, exog=exog, lag_value=lags)
    results_gb = forecast_and_evaluate_gradient_boosting(df_arg=df, exog=exog, lag_value=lags)
    results_xgb = forecast_and_evaluate_xgboost(df_arg=df, exog=exog, lag_value=lags)
    results_dt = forecast_and_evaluate_decision_tree(df_arg=df, exog=exog, lag_value=lags)
    results_lasso = forecast_and_evaluate_lasso(df_arg=df, exog=exog, lag_value=lags)
    results_enr = forecast_and_evaluate_elastic_net(df_arg=df, exog=exog, lag_value=lags)
    results_svr = forecast_and_evaluate_svr(df_arg=df, exog=exog, lag_value=lags)
    results_knn = forecast_and_evaluate_knn(df_arg=df, exog=exog, lag_value=lags)
    
    
    csv_mae = 'evaluations/mae.csv'
    csv_mape = 'evaluations/mape.csv'
    csv_mse = 'evaluations/mse.csv'
    csv_rmse = 'evaluations/rmse.csv'
    
    new_row_mae = pd.DataFrame([[filename,results_ridge['mae'],results_rf['mae'],results_lr['mae'],results_gb['mae'],results_xgb['mae'],results_dt['mae'],results_lasso['mae'],results_enr['mae'],results_svr['mae'], results_knn['mae']]], columns=['fname','ridge','rf','lr','gb','xgb','dt','lasso','enr','svr','knn'])
    new_row_mape = pd.DataFrame([[filename,results_ridge['mape'],results_rf['mape'],results_lr['mape'],results_gb['mape'],results_xgb['mape'],results_dt['mape'],results_lasso['mape'],results_enr['mape'],results_svr['mape'], results_knn['mape']]], columns=['fname','ridge','rf','lr','gb','xgb','dt','lasso','enr','svr','knn'])
    new_row_mse = pd.DataFrame([[filename,results_ridge['mse'],results_rf['mse'],results_lr['mse'],results_gb['mse'],results_xgb['mse'],results_dt['mse'],results_lasso['mse'],results_enr['mse'],results_svr['mse'], results_knn['mse']]], columns=['fname','ridge','rf','lr','gb','xgb','dt','lasso','enr','svr','knn'])
    new_row_rmse = pd.DataFrame([[filename,results_ridge['rmse'],results_rf['rmse'],results_lr['rmse'],results_gb['rmse'],results_xgb['rmse'],results_dt['rmse'],results_lasso['rmse'],results_enr['rmse'],results_svr['rmse'], results_knn['rmse']]], columns=['fname','ridge','rf','lr','gb','xgb','dt','lasso','enr','svr','knn'])
    
    
    new_row_mae.to_csv(csv_mae,  mode='a', header=False, index=False, lineterminator="\n")
    new_row_mape.to_csv(csv_mape, mode='a', header=False, index=False, lineterminator="\n")
    new_row_mse.to_csv(csv_mse, mode='a', header=False, index=False, lineterminator="\n")
    new_row_rmse.to_csv(csv_rmse, mode='a', header=False, index=False, lineterminator="\n")
    
    
    
    