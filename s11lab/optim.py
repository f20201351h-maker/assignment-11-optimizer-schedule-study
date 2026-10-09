"""
A hand-written AdamW whose only reason to exist is the `bias_correction` switch.

With bias_correction=True it performs the same arithmetic, in the same order, as torch.optim.AdamW's
single-tensor path (torch/optim/adam.py, _single_tensor_adam with decoupled weight decay):

    p      <- p * (1 - lr * wd)                       decoupled decay, applied first
    m      <- beta1 * m + (1 - beta1) * g
    v      <- beta2 * v + (1 - beta2) * g^2
    denom  <- sqrt(v) / sqrt(1 - beta2^t) + eps
    p      <- p - (lr / (1 - beta1^t)) * m / denom

which is algebraically lr * m_hat / (sqrt(v_hat) + eps). With bias_correction=False both (1 - beta^t) factors are
replaced by 1, so the step is lr * m / (sqrt(v) + eps). tests/test_optim.py checks the True path against PyTorch.
"""
import torch


class ManualAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, bias_correction=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd, bc = group["lr"], group["eps"], group["weight_decay"], group["bias_correction"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if bc:
                    bc1 = 1 - b1 ** t
                    bc2_sqrt = (1 - b2 ** t) ** 0.5
                    denom = (v.sqrt() / bc2_sqrt).add_(eps)
                    p.addcdiv_(m, denom, value=-lr / bc1)
                else:
                    denom = v.sqrt().add_(eps)
                    p.addcdiv_(m, denom, value=-lr)
        return None
