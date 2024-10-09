import pandas as pd
from sklearn.svm import SVR
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler


def forecast_and_evaluate_svr(df_arg, exog, lag_value):
    """
    Function to perform time series forecasting using an SVR model,
    optimize hyperparameters using random search, and evaluate the model using backtesting.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index and a single column of time series data.

    Returns:
    dict: Dictionary containing the best hyperparameters and evaluation metrics (MAE, MAPE, MSE, RMSE).
    """
    
    
    print("Evaluating SVR.........................................")
    
    
    df = df_arg.copy(deep=True)
    df = df.reset_index()
    df = df.drop(df.columns[0], axis=1)

    forecaster = ForecasterAutoreg(
        regressor=SVR(),
        lags=lag_value,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Define parameter grid for SVR
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],  # Only applicable for poly kernel
    }

    # Perform random search to find the best hyperparameters
    results_random_search = random_search_forecaster(
        forecaster=forecaster,
        y=df.iloc[:, 0],  # The column of time series data
        param_distributions=param_grid,
        steps=10,  
        exog=exog,
        n_iter=10,  
        metric='mean_squared_error', 
        initial_train_size=int(len(df) * 0.8),  # Use 80% for training, rest for validation
        fixed_train_size=False,  
        return_best=True,  # Return the best parameter set
        random_state=123
    )
    print("done random search-------------------------------------------")
    best_params = results_random_search.iloc[0]['params']

    # Recreate the forecaster with the best parameters
    forecaster = ForecasterAutoreg(
        regressor=SVR(**best_params),
        lags=lag_value,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Backtest the model
    backtest_metric, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=df.iloc[:, 0],
        exog=exog,
        initial_train_size=int(len(df) * 0.8),  # 80% train size
        fixed_train_size=False,  
        steps=10,  
        metric='mean_squared_error',
        verbose=True
    )

    y_true = df.iloc[int(len(df) * 0.8):, 0]  # The actual values from the test set
    mae = mean_absolute_error(y_true, predictions)
    mape_val = mean_absolute_percentage_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)

    # Print evaluation metrics
    print(f"MAE: {mae}")
    print(f"MAPE: {mape_val}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # Return results as a dictionary
    return {
        'results_random_search': results_random_search,
        'best_params': best_params,
        'mae': mae,
        'mape': mape_val,
        'mse': mse,
        'rmse': rmse
    }

