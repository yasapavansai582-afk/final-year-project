"""
train.py — Train CapsNet and/or baseline models for TCSP-PC.

Usage:
    python train.py                        # trains all models
    python train.py --model capsnet        # trains CapsNet only
    python train.py --model all --epochs 50
"""

import argparse
import os
import time
import numpy as np

from preprocessing import load_processed
from capsnet import CapsNetClassifier
from baselines import VEQMClassifier, MPTCPRPClassifier, MWISClassifier, CONVBiLSTMClassifier as ConvBiLSTMClassifier

SAVE_DIR = "models/saved"


def train_capsnet(X_train, y_train, X_val, y_val, epochs=30):
    print("\n" + "=" * 60)
    print("  Training: CapsNet (TCSP-PC)")
    print("=" * 60)
    model = CapsNetClassifier(
        n_primary_caps=8,
        primary_cap_dim=8,
        n_emission_caps=4,
        emission_cap_dim=4,
        n_routing=3,
        lr=0.01,
        epochs=epochs,
        batch_size=256,
        seed=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    elapsed = time.time() - t0
    acc = model.score(X_val, y_val)
    print(f"  ✓ Val Accuracy: {acc:.4f}  |  Training time: {elapsed:.1f}s")
    model.save(os.path.join(SAVE_DIR, "capsnet.pkl"))
    return model


def train_veqm(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("  Training: VEQM Baseline")
    print("=" * 60)
    model = VEQMClassifier(C=1.0, max_iter=500, seed=42)
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    acc = model.score(X_val, y_val)
    print(f"  ✓ Val Accuracy: {acc:.4f}  |  Training time: {elapsed:.1f}s")
    model.save(os.path.join(SAVE_DIR, "veqm.pkl"))
    return model


def train_mptcprp(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("  Training: MPTC-PRP Baseline")
    print("=" * 60)
    model = MPTCPRPClassifier(n_estimators=150, max_depth=4, lr=0.05, seed=42)
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    acc = model.score(X_val, y_val)
    print(f"  ✓ Val Accuracy: {acc:.4f}  |  Training time: {elapsed:.1f}s")
    model.save(os.path.join(SAVE_DIR, "mptcprp.pkl"))
    return model


def train_mwis(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("  Training: MWIS Baseline")
    print("=" * 60)
    model = MWISClassifier(n_estimators=100, seed=42)
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    acc = model.score(X_val, y_val)
    print(f"  ✓ Val Accuracy: {acc:.4f}  |  Training time: {elapsed:.1f}s")
    model.save(os.path.join(SAVE_DIR, "mwis.pkl"))
    return model


def train_convbilstm(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("  Training: CONV-BI-LSTM Baseline")
    print("=" * 60)
    model = ConvBiLSTMClassifier(hidden_dim=64, epochs=20, lr=0.01, seed=42)
    t0 = time.time()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    elapsed = time.time() - t0
    acc = model.score(X_val, y_val)
    print(f"  ✓ Val Accuracy: {acc:.4f}  |  Training time: {elapsed:.1f}s")
    model.save(os.path.join(SAVE_DIR, "convbilstm.pkl"))
    return model


def main():
    parser = argparse.ArgumentParser(description="Train TCSP-PC models")
    parser.add_argument("--model",  default="all",
                        choices=["capsnet", "veqm", "mptcprp", "mwis", "convbilstm", "all"],
                        help="Which model to train (default: all)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs for CapsNet / CONV-BI-LSTM (default: 30)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TCSP-PC — Model Training")
    print("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = load_processed()
    print(f"\n  Features: {len(feature_names)}")
    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    m = args.model
    if m in ("capsnet", "all"):
        train_capsnet(X_train, y_train, X_val, y_val, epochs=args.epochs)
    if m in ("veqm", "all"):
        train_veqm(X_train, y_train, X_val, y_val)
    if m in ("mptcprp", "all"):
        train_mptcprp(X_train, y_train, X_val, y_val)
    if m in ("mwis", "all"):
        train_mwis(X_train, y_train, X_val, y_val)
    if m in ("convbilstm", "all"):
        train_convbilstm(X_train, y_train, X_val, y_val, )

    print("\n" + "=" * 60)
    print("  All requested models trained and saved.")
    print("  Run: python evaluate.py   to compare results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()