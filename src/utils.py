import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def save_metrics(y_true, y_pred, model_name, metrics_dict):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics_dict[model_name] = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
    return metrics_dict

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def save_model(model, path):
    joblib.dump(model, path)
