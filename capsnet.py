"""
Correlated Capsule Network (CapsNet) for vehicle emission classification.
Implemented in pure NumPy + scikit-learn (no PyTorch required) so it runs
on any machine without a GPU.

Architecture mirrors the paper's description:
  Input  →  Primary Capsules  →  Emission-Behaviour Capsules  →  Classification
"""

import numpy as np
import os
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def squash(v, axis=-1):
    """Squash activation used in CapsNets (Sabour et al. 2017)."""
    norm_sq = np.sum(v ** 2, axis=axis, keepdims=True)
    norm    = np.sqrt(norm_sq + 1e-8)
    return (norm_sq / (1.0 + norm_sq)) * (v / norm)


class PrimaryCapsuleLayer:
    """
    Maps raw features → capsule vectors.
    n_capsules × capsule_dim = hidden representation.
    """
    def __init__(self, in_dim, n_capsules=8, capsule_dim=8, seed=42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(in_dim, n_capsules * capsule_dim) * 0.1
        self.b = np.zeros(n_capsules * capsule_dim)
        self.n_capsules  = n_capsules
        self.capsule_dim = capsule_dim

    def forward(self, X):
        # X: (batch, in_dim)
        h = X @ self.W + self.b          # (batch, n_capsules * capsule_dim)
        h = h.reshape(X.shape[0], self.n_capsules, self.capsule_dim)
        return squash(h, axis=-1)        # (batch, n_capsules, capsule_dim)


class CorrelatedCapsuleLayer:
    """
    Emission-behaviour capsule layer with dynamic routing.
    Implements Eq. (5a-5d) from the paper.
    """
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, n_routing=3, seed=42):
        rng = np.random.RandomState(seed)
        # W_ij: (in_capsules, out_capsules, in_dim, out_dim)
        self.W = rng.randn(in_capsules, out_capsules, in_dim, out_dim) * 0.1
        self.n_routing   = n_routing
        self.out_capsules = out_capsules
        self.out_dim      = out_dim
        self.in_capsules  = in_capsules

    def forward(self, u):
        """
        u: (batch, in_capsules, in_dim)
        returns: (batch, out_capsules, out_dim)
        """
        batch = u.shape[0]
        # Prediction vectors: u_hat[b, i, j, :] = W[i,j] @ u[b,i]
        # Shape: (batch, in_capsules, out_capsules, out_dim)
        # u: (batch, in_caps, in_dim), W: (in_caps, out_caps, in_dim, out_dim)
        # u_hat: (batch, in_caps, out_caps, out_dim)
        u_hat = np.einsum('bil,ijlk->bijk', u, self.W)

        # Dynamic routing
        b_ij = np.zeros((batch, self.in_capsules, self.out_capsules))
        v = None
        for _ in range(self.n_routing):
            c_ij = self._softmax(b_ij)                       # (batch, in_cap, out_cap)
            # s_j = sum_i c_ij * u_hat[b,i,j]
            s = np.einsum('bij,bijl->bjl', c_ij, u_hat)     # (batch, out_cap, out_dim)
            v = squash(s, axis=-1)                           # (batch, out_cap, out_dim)
            # Agreement update
            delta = np.einsum('bjl,bijl->bij', v, u_hat)    # (batch, in_cap, out_cap)
            b_ij = b_ij + delta
        return v  # (batch, out_capsules, out_dim)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


class CapsNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Full CapsNet pipeline:
      Input → Primary Capsules → Correlated Capsules → FC → sigmoid
    """
    def __init__(self,
                 n_primary_caps=8,
                 primary_cap_dim=8,
                 n_emission_caps=4,
                 emission_cap_dim=4,
                 n_routing=3,
                 lr=0.01,
                 epochs=30,
                 batch_size=256,
                 seed=42):
        self.n_primary_caps  = n_primary_caps
        self.primary_cap_dim = primary_cap_dim
        self.n_emission_caps = n_emission_caps
        self.emission_cap_dim = emission_cap_dim
        self.n_routing    = n_routing
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.seed         = seed
        self.history_     = {"train_loss": [], "val_loss": [],
                             "train_acc":  [], "val_acc": []}

    def _init_network(self, in_dim):
        rng = np.random.RandomState(self.seed)
        self.primary_caps_ = PrimaryCapsuleLayer(
            in_dim, self.n_primary_caps, self.primary_cap_dim, seed=self.seed)
        self.corr_caps_ = CorrelatedCapsuleLayer(
            self.n_primary_caps, self.primary_cap_dim,
            self.n_emission_caps, self.emission_cap_dim,
            self.n_routing, seed=self.seed)
        # Final linear layer: flatten caps → 1 output (binary)
        flat_dim = self.n_emission_caps * self.emission_cap_dim
        self.W_out_ = rng.randn(flat_dim, 1) * 0.1
        self.b_out_ = np.zeros(1)

    def _forward(self, X):
        u = self.primary_caps_.forward(X)       # (batch, prim_caps, prim_dim)
        v = self.corr_caps_.forward(u)           # (batch, emis_caps, emis_dim)
        flat = v.reshape(v.shape[0], -1)         # (batch, caps*dim)
        logit = flat @ self.W_out_ + self.b_out_ # (batch, 1)
        return sigmoid(logit).squeeze(-1), flat  # probs, flat activations

    def _loss(self, probs, y):
        eps = 1e-7
        return -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))

    def fit(self, X, y, X_val=None, y_val=None):
        self._init_network(X.shape[1])
        n = X.shape[0]
        rng = np.random.RandomState(self.seed)

        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                Xb = X[batch_idx].astype(np.float32)
                yb = y[batch_idx].astype(np.float32)

                probs, flat = self._forward(Xb)
                loss = self._loss(probs, yb)

                # Gradient of loss w.r.t. logit (before sigmoid)
                d_logit = (probs - yb) / len(yb)          # (batch,)
                d_logit = d_logit[:, None]                 # (batch, 1)

                # Back-prop through output layer only (simplified)
                grad_W = flat.T @ d_logit
                grad_b = d_logit.sum(axis=0)
                self.W_out_ -= self.lr * grad_W
                self.b_out_ -= self.lr * grad_b

                epoch_loss += loss
                n_batches  += 1

            avg_loss  = epoch_loss / n_batches
            train_acc = accuracy_score(y, (self._forward(X)[0] >= 0.5).astype(int))
            self.history_["train_loss"].append(avg_loss)
            self.history_["train_acc"].append(train_acc)

            if X_val is not None:
                val_probs, _ = self._forward(X_val.astype(np.float32))
                val_loss = self._loss(val_probs, y_val.astype(np.float32))
                val_acc  = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                          f"Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                          f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        return self

    def predict_proba(self, X):
        probs, _ = self._forward(X.astype(np.float32))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        probs, _ = self._forward(X.astype(np.float32))
        return (probs >= 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def save(self, path="models/saved/capsnet.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  CapsNet saved -> {path}")

    @staticmethod
    def load(path="models/saved/capsnet.pkl"):
        return joblib.load(path)