import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import sys

def preprocess(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Select relevant columns
    features = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
    target = 'cpu_usage'

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing
    numeric_features = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes']
    categorical_features = ['controller_kind']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Save splits
    joblib.dump((X_train, X_test, y_train, y_test, preprocessor), output_path)

if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2])
