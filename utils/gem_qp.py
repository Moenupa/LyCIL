import torch

def project_to_cone(g: torch.Tensor, G: torch.Tensor, max_iter: int = 100, lr: float = 0.1, tol: float = 1e-6):
    """Solve: min ||g - g_tilde||^2 s.t. G @ g_tilde >= 0 via dual PGD.

    Dual: min_{lambda>=0} 0.5 * lambda^T (G G^T) lambda + (G g)^T lambda
    Then g_tilde = g + G^T lambda*.
    """
    if G.numel() == 0 or G.shape[0] == 0:
        return g
    device = g.device
    Q = G @ G.t()   # (k,k)
    p = G @ g       # (k,)
    lam = torch.zeros(Q.size(0), device=device)
    # Heuristic step size using Lipschitz estimate (largest eigenvalue of Q) bounded
    L = torch.linalg.norm(Q, ord=2).clamp(min=1.0).item()
    step = min(lr, 1.0 / L)
    for _ in range(max_iter):
        grad = Q @ lam + p              # gradient of dual
        lam_prev = lam
        lam = lam - step * grad
        lam = torch.clamp(lam, min=0.0) # projection onto nonnegatives
        if torch.norm(lam - lam_prev).item() < tol:
            break
    g_tilde = g + G.t() @ lam
    return g_tilde
