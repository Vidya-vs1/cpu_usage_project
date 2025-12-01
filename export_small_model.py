import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# 1️⃣ CHANGE THIS PATH IF NEEDED
DATA_PATH = Path("data/cpu_data.csv")  # or data/cpu_data.csv if that's your file
# MODEL_PATH = Path("models/model_RandomForest_small.joblib")
# 2️⃣ FEATURES & TARGET
FEATURE_COLS = [
    "cpu_request",
    "mem_request",
    "cpu_limit",
    "mem_limit",
    "runtime_minutes",
    "controller_kind",
]
TARGET_COL = "cpu_usage"

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Handle missing controller_kind by filling "Unknown"
df["controller_kind"] = df["controller_kind"].fillna("Unknown")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

numeric_features = ["cpu_request", "mem_request", "cpu_limit", "mem_limit", "runtime_minutes"]
categorical_features = ["controller_kind"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 3️⃣ SMALLER RANDOM FOREST (LIGHTER MODEL)
rf = RandomForestRegressor(
    n_estimators=80,      # fewer trees
    max_depth=20,        # limit tree depth
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf),
])

print("Training smaller RandomForest model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# 4️⃣ SAVE COMPRESSED MODEL
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

OUT_PATH = models_dir / "model_RandomForest_small.joblib"
print(f"Saving compressed model to: {OUT_PATH}")

joblib.dump(pipeline, OUT_PATH, compress=3)

print("Done! Now check the file size of:", OUT_PATH)
