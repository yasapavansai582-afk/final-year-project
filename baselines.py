"""
Baseline models for comparison with TCSP-PC:
  1. VEQM  — Vehicle Emission Quantification Method (Yu et al. [28])
  2. MPTC-PRP — Multi-Path Traffic-Covering Pollution Routing Process (Hosseini-Motlagh et al. [24])
  3. MWIS  — Maximum Weight Independent Set (Bai et al. [29])
  4. CONV-BI-LSTM — Convolutional Bidirectional LSTM (Bilotta et al. [37])

All baselines are implemented as scikit-learn-compatible estimators
using only NumPy + scikit-learn so no GPU is required.
"""

import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. VEQM  ─ Vehicle Emission Quantification Method
# ─────────────────────────────────────────────────────────────────────────────
class VEQMClassifier(BaseEstimator, ClassifierMixin):
    """
    Implements VEQM logic: quantifies per-vehicle emission index, then
    applies a logistic regression boundary for eco / polluting split.
    Reference: Yu et al. [28] — J. Cleaner Prod., 328, 2021.
    """

    def __init__(self, C=1.0, max_iter=500, seed=42):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def _emission_index(self, X):
        """
        Compute emission index from raw features.
        Uses emission_rate (col 3) weighted by distance_km (col 2) and
        vehicle_density (col 4).  All columns after StandardScaler.
        """
        emission_rate    = X[:, 3]
        distance_km      = X[:, 2]
        vehicle_density  = X[:, 4]
        travel_time      = X[:, 5]

        # VEQM emission index (Eq-based weighted sum)
        ei = emission_rate * (1 + 0.1 * np.abs(distance_km)) * \
             (1 + 0.05 * np.maximum(vehicle_density, 0)) / \
             (1 + 0.02 * np.maximum(travel_time, 0.1))
        return ei[:, None]   # column vector

    def fit(self, X, y, **kwargs):
        ei = self._emission_index(X)
        X_aug = np.hstack([X, ei])
        self.lr_ = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                      random_state=self.seed, solver="lbfgs")
        self.lr_.fit(X_aug, y)
        self.history_ = {"train_acc": [accuracy_score(y, self.lr_.predict(X_aug))]}
        return self

    def predict(self, X):
        return self.lr_.predict(np.hstack([X, self._emission_index(X)]))

    def predict_proba(self, X):
        return self.lr_.predict_proba(np.hstack([X, self._emission_index(X)]))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def save(self, path="models/saved/veqm.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  VEQM saved -> {path}")

    @staticmethod
    def load(path="models/saved/veqm.pkl"):
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MPTC-PRP  ─ Multi-Path Traffic-Covering Pollution Routing Process
# ─────────────────────────────────────────────────────────────────────────────
class MPTCPRPClassifier(BaseEstimator, ClassifierMixin):
    """
    Implements MPTC-PRP: gradient boosting that incorporates multi-path
    routing cost as an additional feature (pollution × distance matrix).
    Reference: Hosseini-Motlagh et al. [24] — Comput. Ind. Eng., 173, 2022.
    """

    def __init__(self, n_estimators=150, max_depth=4, lr=0.05, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.seed = seed

    def _routing_feature(self, X):
        """
        Simulates multi-path routing cost:
        routing_cost = emission_rate × distance_km × congestion_index
        congestion_index = vehicle_density / max(travel_time, 0.1)
        """
        emission_rate   = X[:, 3]
        distance_km     = X[:, 2]
        vehicle_density = X[:, 4]
        travel_time     = X[:, 5]

        congestion = np.maximum(vehicle_density, 0) / \
                     np.maximum(travel_time, 0.1)
        routing_cost = emission_rate * np.maximum(distance_km, 0) * \
                       (1 + 0.05 * congestion)
        return routing_cost[:, None]

    def fit(self, X, y, **kwargs):
        rc = self._routing_feature(X)
        X_aug = np.hstack([X, rc])
        self.gb_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr,
            random_state=self.seed,
        )
        self.gb_.fit(X_aug, y)
        self.history_ = {"train_acc": [accuracy_score(y, self.gb_.predict(X_aug))]}
        return self

    def predict(self, X):
        return self.gb_.predict(np.hstack([X, self._routing_feature(X)]))

    def predict_proba(self, X):
        return self.gb_.predict_proba(np.hstack([X, self._routing_feature(X)]))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def save(self, path="models/saved/mptcprp.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  MPTC-PRP saved -> {path}")

    @staticmethod
    def load(path="models/saved/mptcprp.pkl"):
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MWIS  ─ Maximum Weight Independent Set
# ─────────────────────────────────────────────────────────────────────────────
class MWISClassifier(BaseEstimator, ClassifierMixin):
    """
    MWIS-inspired classifier: builds a conflict graph between vehicles
    based on emission proximity, then uses a greedy MIS heuristic score
    as a feature fed into Random Forest.
    Reference: Bai & Bai [29] — IEEE Access, 9, 2021.
    """

    def __init__(self, n_estimators=100, conflict_threshold=0.5, seed=42):
        self.n_estimators = n_estimators
        self.conflict_threshold = conflict_threshold
        self.seed = seed

    def _mwis_score(self, X):
        """
        Computes a per-sample conflict score:
        how much a vehicle 'conflicts' with an ideally clean traffic set.
        """
        emission_rate   = X[:, 3]
        vehicle_density = X[:, 4]
        distance_km     = X[:, 2]

        # Normalised weighted conflict score
        norm_em  = emission_rate / (np.max(np.abs(emission_rate)) + 1e-8)
        norm_den = vehicle_density / (np.max(np.abs(vehicle_density)) + 1e-8)
        score    = norm_em * (1 + 0.3 * norm_den) * \
                   np.sqrt(np.maximum(distance_km, 0) + 1)
        return score[:, None]

    def fit(self, X, y, **kwargs):
        ms = self._mwis_score(X)
        X_aug = np.hstack([X, ms])
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.rf_.fit(X_aug, y)
        self.history_ = {"train_acc": [accuracy_score(y, self.rf_.predict(X_aug))]}
        return self

    def predict(self, X):
        return self.rf_.predict(np.hstack([X, self._mwis_score(X)]))

    def predict_proba(self, X):
        return self.rf_.predict_proba(np.hstack([X, self._mwis_score(X)]))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def save(self, path="models/saved/mwis.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  MWIS saved -> {path}")

    @staticmethod
    def load(path="models/saved/mwis.pkl"):
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONV-BI-LSTM  ─ Convolutional Bidirectional LSTM (NumPy version)
# ─────────────────────────────────────────────────────────────────────────────
def _tanh(x):
    return np.tanh(np.clip(x, -30, 30))

def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class SimpleLSTMCell:
    """Minimal LSTM cell implemented in NumPy."""

    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.RandomState(seed)
        scale = 0.1
        dim = input_dim + hidden_dim
        self.Wf = rng.randn(dim, hidden_dim) * scale
        self.Wi = rng.randn(dim, hidden_dim) * scale
        self.Wc = rng.randn(dim, hidden_dim) * scale
        self.Wo = rng.randn(dim, hidden_dim) * scale
        self.bf = np.ones(hidden_dim) * 0.1
        self.bi = np.zeros(hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward_seq(self, X):
        """X: (batch, seq_len, input_dim) → outputs: (batch, seq_len, hidden_dim)"""
        batch, seq, _ = X.shape
        h = np.zeros((batch, self.hidden_dim))
        c = np.zeros((batch, self.hidden_dim))
        outputs = []
        for t in range(seq):
            xt  = X[:, t, :]
            xh  = np.concatenate([xt, h], axis=1)
            f   = _sigmoid_np(xh @ self.Wf + self.bf)
            i   = _sigmoid_np(xh @ self.Wi + self.bi)
            ct  = _tanh(xh @ self.Wc + self.bc)
            o   = _sigmoid_np(xh @ self.Wo + self.bo)
            c   = f * c + i * ct
            h   = o * _tanh(c)
            outputs.append(h[:, None, :])
        return np.concatenate(outputs, axis=1)


class CONVBiLSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    Simplified Conv + Bidirectional LSTM for emission classification.
    Conv layer: 1-D convolution over feature dimension (temporal window).
    Bi-LSTM: forward + backward LSTM, concatenated.
    Reference: Bilotta et al. [37] — IEEE Access, 10, 2022.
    """

    def __init__(self, hidden_dim=32, epochs=30, lr=0.005,
                 batch_size=256, seed=42):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.history_ = {"train_loss": [], "val_loss": [],
                          "train_acc":  [], "val_acc": []}

    def _conv1d(self, X, kernel_size=3):
        """
        Simple 1-D average pooling as a proxy for a convolutional layer.
        X: (batch, features) → (batch, features, 1) treated as seq.
        """
        # Treat each feature as a "time step" of length 1
        # Stack sliding windows of size kernel_size
        n, f = X.shape
        padded = np.pad(X, ((0, 0), (kernel_size // 2, kernel_size // 2)),
                        mode='edge')
        windows = np.stack(
            [padded[:, i:i + f] for i in range(kernel_size)], axis=-1
        )                                     # (batch, features, kernel_size)
        # Conv output: mean across kernel window
        conv_out = windows.mean(axis=-1)      # (batch, features)
        return conv_out[:, :, None]           # (batch, features, 1) = seq of len features

    def _build(self, in_dim):
        rng = np.random.RandomState(self.seed)
        # LSTM cells for forward and backward pass
        self.lstm_fwd_ = SimpleLSTMCell(1, self.hidden_dim, seed=self.seed)
        self.lstm_bwd_ = SimpleLSTMCell(1, self.hidden_dim, seed=self.seed + 1)
        # Output layer: 2*hidden → 1
        flat = 2 * self.hidden_dim
        self.W_out_ = rng.randn(flat, 1) * 0.1
        self.b_out_ = np.zeros(1)

    def _forward(self, X):
        conv_seq = self._conv1d(X)           # (batch, seq=features, 1)
        fwd = self.lstm_fwd_.forward_seq(conv_seq)   # (batch, seq, h)
        bwd = self.lstm_bwd_.forward_seq(conv_seq[:, ::-1, :])  # reversed

        # Use last hidden states
        h_fwd = fwd[:, -1, :]
        h_bwd = bwd[:, -1, :]
        combined = np.concatenate([h_fwd, h_bwd], axis=1)  # (batch, 2h)

        logit = combined @ self.W_out_ + self.b_out_       # (batch, 1)
        prob  = _sigmoid_np(logit).squeeze(-1)
        return prob, combined

    def _bce(self, probs, y):
        eps = 1e-7
        return -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))

    def fit(self, X, y, X_val=None, y_val=None):
        self._build(X.shape[1])
        n   = X.shape[0]
        rng = np.random.RandomState(self.seed)

        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                bi  = idx[start:start + self.batch_size]
                Xb  = X[bi].astype(np.float32)
                yb  = y[bi].astype(np.float32)
                probs, combined = self._forward(Xb)
                loss = self._bce(probs, yb)
                # Gradient w.r.t. output layer only
                d = (probs - yb)[:, None] / len(yb)
                self.W_out_ -= self.lr * (combined.T @ d)
                self.b_out_ -= self.lr * d.sum(axis=0)
                ep_loss += loss
                n_batches += 1

            avg = ep_loss / n_batches
            tr_acc = accuracy_score(y, (self._forward(X.astype(np.float32))[0] >= 0.5).astype(int))
            self.history_["train_loss"].append(avg)
            self.history_["train_acc"].append(tr_acc)

            if X_val is not None:
                vp, _ = self._forward(X_val.astype(np.float32))
                vl = self._bce(vp, y_val.astype(np.float32))
                va = accuracy_score(y_val, (vp >= 0.5).astype(int))
                self.history_["val_loss"].append(vl)
                self.history_["val_acc"].append(va)
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                          f"Loss: {avg:.4f} | Val: {vl:.4f} | "
                          f"Tr Acc: {tr_acc:.4f} | Val Acc: {va:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                          f"Loss: {avg:.4f} | Tr Acc: {tr_acc:.4f}")
        return self

    def predict_proba(self, X):
        probs, _ = self._forward(X.astype(np.float32))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        probs, _ = self._forward(X.astype(np.float32))
        return (probs >= 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def save(self, path="models/saved/conv_bilstm.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  CONV-BI-LSTM saved -> {path}")

    @staticmethod
    def load(path="models/saved/conv_bilstm.pkl"):
        return joblib.load(path)