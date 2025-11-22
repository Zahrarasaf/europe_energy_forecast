import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def evaluate_predictions(y_true, y_pred):
    """
    Compute RMSE and MAPE for model performance evaluation.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": rmse, "mape": mape}
