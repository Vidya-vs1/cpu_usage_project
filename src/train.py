import joblib
import json
import sys
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

def train(input_path, metrics_path, model_path):
    # Load data
    X_train, X_test, y_train, y_test, preprocessor = joblib.load(input_path)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel="rbf")
    }

    metrics = {}

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        metrics[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

        # Save model
        joblib.dump(pipe, f"{model_path}_{name}.joblib")

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], sys.argv[3])
