"""ANFIS — Adaptive Neuro-Fuzzy Inference System (Jang 1993).

Architecture (thesis Ch 4.4):
- 5-layer ANFIS
- Default in production runs: 5 inputs × 3 Gaussian MFs each = 243 rules (3^5).
  The class accepts any input_dim; the runtime trainer infers it from
  X_train.shape[1], so the default below is documentary only.
- Premise (MF centers/sigmas): k-means initialization on training data,
  seeded by the per-experiment seed for genuine seed-to-seed variability.
- Consequent: ridge least-squares solution (lambda = 1e-3) — closed-form,
  not gradient.

For binary classification, the output is squashed with sigmoid.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class ANFIS(nn.Module):
    def __init__(self, input_dim: int = 4, n_mfs: int = 3, ridge_lambda: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** input_dim
        self.ridge = ridge_lambda
        # Premise parameters (Gaussian MFs): centers and sigmas per (feature, mf)
        self.centers = nn.Parameter(torch.zeros(input_dim, n_mfs), requires_grad=False)
        self.sigmas = nn.Parameter(torch.ones(input_dim, n_mfs), requires_grad=False)
        # Consequent: per rule, a linear function of inputs: w_r = a_r^T x + b_r
        # Solved by ridge LSE on training data; stored as parameter.
        self.consequent = nn.Parameter(torch.zeros(self.n_rules, input_dim + 1), requires_grad=False)
        # Precomputed rule indexing: rule r maps to (mf_for_feature_0, ..., mf_for_feature_F-1)
        self.register_buffer('rule_mfs', self._build_rule_table())

    def _build_rule_table(self):
        ranges = [list(range(self.n_mfs)) for _ in range(self.input_dim)]
        import itertools
        return torch.tensor(list(itertools.product(*ranges)), dtype=torch.long)

    def fit_premise(self, X: np.ndarray, seed: int = 0):
        """Initialize Gaussian MF centers using k-means PER FEATURE.

        The k-means seed flows through from the experiment seed, so that
        seed-to-seed variation in ANFIS premise initialisation is genuine.
        """
        for f in range(self.input_dim):
            xf = X[:, f].reshape(-1, 1)
            km = KMeans(n_clusters=self.n_mfs, n_init=5, random_state=seed).fit(xf)
            centers = np.sort(km.cluster_centers_.flatten())
            self.centers.data[f] = torch.tensor(centers, dtype=torch.float32)
            # sigma = average gap between centers, with floor
            if self.n_mfs > 1:
                gaps = np.diff(centers)
                sig = max(gaps.mean(), 1e-3)
            else:
                sig = max(xf.std(), 1e-3)
            self.sigmas.data[f] = torch.tensor([sig] * self.n_mfs, dtype=torch.float32)

    def fit_consequent(self, X: np.ndarray, y: np.ndarray):
        """Compute rule firing strengths, then ridge LSE for consequent parameters.
        
        For binary classification: targets are {0, 1} or scaled to {-1, +1}.
        We use the raw {0,1} and apply sigmoid at inference.
        """
        Xt = torch.tensor(X, dtype=torch.float32)
        # Compute normalized rule firing strengths: (N, R)
        firing = self._firing_strengths(Xt)
        firing_norm = firing / (firing.sum(dim=1, keepdim=True) + 1e-9)
        # Design matrix for consequent: for each (sample, rule), the (firing_r * [x | 1])
        N = X.shape[0]
        x_aug = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N, F+1)
        f_np = firing_norm.numpy()
        # Stack columns: for each rule, weight x_aug by f_np[:, r] -> total cols = R*(F+1)
        cols = []
        for r in range(self.n_rules):
            cols.append(x_aug * f_np[:, [r]])
        A = np.concatenate(cols, axis=1)  # (N, R*(F+1))
        # Ridge LSE: solve (A^T A + lambda I) w = A^T y
        AtA = A.T @ A
        Aty = A.T @ y
        w = np.linalg.solve(AtA + self.ridge * np.eye(AtA.shape[0]), Aty)
        # Reshape to (R, F+1)
        self.consequent.data = torch.tensor(w.reshape(self.n_rules, self.input_dim + 1),
                                            dtype=torch.float32)

    def _firing_strengths(self, X: torch.Tensor) -> torch.Tensor:
        """Compute rule firing strengths via product of MF memberships.
        
        X: (N, F). Returns (N, R) where R = n_mfs**F.
        """
        # MF activations for each (sample, feature, mf): Gaussian
        # (N, F, M) where M = n_mfs
        N, F = X.shape
        diffs = X.unsqueeze(-1) - self.centers.unsqueeze(0)  # (N, F, M)
        memb = torch.exp(-0.5 * (diffs / (self.sigmas.unsqueeze(0) + 1e-9)) ** 2)
        # For each rule, product over features of memb[:, f, mf_table[r, f]]
        rule_mfs = self.rule_mfs  # (R, F)
        R = rule_mfs.shape[0]
        firing = torch.ones(N, R, device=X.device)
        for f in range(F):
            mf_idx = rule_mfs[:, f]  # (R,)
            sel_f = memb[:, f, :].index_select(1, mf_idx)  # (N, R)
            firing = firing * sel_f
        return firing

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, F). Returns logits (N,)."""
        firing = self._firing_strengths(X)
        firing_norm = firing / (firing.sum(dim=1, keepdim=True) + 1e-9)
        # Consequent: (R, F+1)
        x_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)  # (N, F+1)
        # Each rule's local output: x_aug @ consequent[r, :]
        rule_outputs = x_aug @ self.consequent.t()  # (N, R)
        return (firing_norm * rule_outputs).sum(dim=1)


def count_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == '__main__':
    model = ANFIS(input_dim=4, n_mfs=3)
    print(f'ANFIS rules: {model.n_rules}')
    print(f'ANFIS total params: {count_params(model)}')
    X = np.random.randn(200, 4).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    model.fit_premise(X)
    model.fit_consequent(X, y)
    pred = torch.sigmoid(model(torch.tensor(X)))
    print(f'Sample preds: {pred[:5].tolist()}')
    print(f'Acc on train: {((pred.numpy() > 0.5) == y).mean():.3f}')
