"""Modern sequence-only Kermut and MLDE classical RF adapters."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .features import one_hot


def mlde_predict(train_sequences, targets, test_sequences, seed, trees=100, jobs=1):
    model = RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=jobs)
    model.fit(one_hot(train_sequences), np.asarray(targets, dtype=float))
    return model.predict(one_hot(test_sequences))


def random_predict(count, seed):
    return np.random.default_rng(seed).permutation(count).astype(float)


def kermut_predict(train_features, targets, test_features, train_prior, test_prior,
                   seed, device="cpu", steps=150, learning_rate=0.03, batch_size=1024):
    import torch
    import gpytorch

    class SequenceKernel(gpytorch.kernels.Kernel):
        is_stationary = True

        def __init__(self, kernel_type="RBF", **kwargs):
            super().__init__(**kwargs)
            if kernel_type != "RBF":
                raise NotImplementedError
            self.base_kernel = gpytorch.kernels.RBFKernel(**kwargs)

        def forward(self, x1, x2, diag=False, **params):
            return self.base_kernel.forward(x1, x2, diag=diag, **params)

    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    dtype = torch.float64
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    normalized_train = (train_features - feature_mean) / feature_scale
    normalized_test = (test_features - feature_mean) / feature_scale

    def tensor(values):
        return torch.as_tensor(values, dtype=dtype, device=device)

    train_input = tensor(normalized_train)
    train_residual = tensor(np.asarray(targets) - np.asarray(train_prior))
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)

    class SequenceGP(gpytorch.models.ExactGP):
        def __init__(self):
            super().__init__(train_input, train_residual, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(SequenceKernel(kernel_type="RBF"))
            self.covar_module.base_kernel.base_kernel.lengthscale = np.sqrt(train_features.shape[1])

        def forward(self, features):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(features),
                                                              self.covar_module(features))

    model = SequenceGP().to(device=device, dtype=dtype)
    model.train()
    likelihood.train()
    objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = -objective(model(train_input), train_residual)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite GP objective at step {step}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    likelihood.eval()
    means = []
    with torch.inference_mode(), gpytorch.settings.fast_pred_var():
        for start in range(0, len(normalized_test), batch_size):
            means.append(model(tensor(normalized_test[start:start + batch_size])).mean.cpu().numpy())
    return np.asarray(test_prior) + np.concatenate(means), losses