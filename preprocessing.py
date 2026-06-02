"""
Preprocessing pipeline — scales features, encodes categoricals,
splits into train/val/test sets, and saves processed arrays.
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "hour", "day_of_week", "distance_km",
    "emission_rate", "vehicle_density", "travel_time_min",
    "city_emission_factor",
]
TARGET_COL = "emission_label"
CAT_COLS   = ["vehicle_type", "city"]


def load_raw(path="data/raw/iot_traffic_data.csv"):
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {path}")
    return df


def encode_categoricals(df):
    df = df.copy()
    le_vtype = LabelEncoder()
    le_city  = LabelEncoder()
    df["vehicle_type_enc"] = le_vtype.fit_transform(df["vehicle_type"])
    df["city_enc"]         = le_city.fit_transform(df["city"])
    return df, le_vtype, le_city


def build_features(df):
    """Return X (numpy array) and y (numpy array)."""
    num_features = FEATURE_COLS + ["vehicle_type_enc", "city_enc"]
    X = df[num_features].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)
    return X, y, num_features


def preprocess(raw_path="data/raw/iot_traffic_data.csv",
               out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    df = load_raw(raw_path)
    df, le_vtype, le_city = encode_categoricals(df)
    X, y, feature_names = build_features(df)

    # Train / Val / Test  =  70 / 15 / 15
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, random_state=42, stratify=y_tv)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save arrays
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

    joblib.dump(scaler,   os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(le_vtype, os.path.join(out_dir, "le_vtype.pkl"))
    joblib.dump(le_city,  os.path.join(out_dir, "le_city.pkl"))
    joblib.dump(feature_names, os.path.join(out_dir, "feature_names.pkl"))

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Scaler + encoders saved to {out_dir}/")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


def load_processed(out_dir="data/processed"):
    X_train = np.load(os.path.join(out_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(out_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(out_dir, "X_test.npy"))
    y_train = np.load(os.path.join(out_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(out_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(out_dir, "y_test.npy"))
    scaler  = joblib.load(os.path.join(out_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(out_dir, "feature_names.pkl"))
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


if __name__ == "__main__":
    preprocess()