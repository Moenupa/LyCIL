import torch
from torch.optim import Optimizer


def _clone_param_spec(params):
    params = list(params)
    if not params:
        return []

    if isinstance(params[0], dict):
        out = []
        for g in params:
            ng = dict(g)
            ng["params"] = list(g["params"])
            out.append(ng)
        return out

    return list(params)


def _as_scalar(x):
    return x.item() if isinstance(x, torch.Tensor) else x


class AdamWThenSGD(Optimizer):
    """
    一个适配 Lightning configure_optimizers() 的包装优化器：

    - 内部用原生 torch.optim.AdamW 和 torch.optim.SGD
    - 外部只暴露一个 optimizer，方便挂同一个 scheduler
    - scheduler 维护的是 outer param_groups 的 lr
    - 每次 step 前，把 outer lr 同步到内部两个 optimizer：
        adamw_lr = outer_lr
        sgd_lr   = outer_lr * (sgd_init_lr / adamw_init_lr)

    这样就实现了：
    - AdamW 和 SGD 各有自己的初始 lr
    - 但两者共享同一条 scheduler 曲线（倍率不同）
    """

    def __init__(
        self,
        params,
        *,
        adamw: dict,
        sgd: dict,
        switch_step: int,
        transfer_momentum: bool = False,
    ):
        if "lr" not in adamw or "lr" not in sgd:
            raise ValueError("Both `adamw` and `sgd` must provide `lr`.")

        outer_params = _clone_param_spec(params)
        adamw_params = _clone_param_spec(params)
        sgd_params = _clone_param_spec(params)

        super().__init__(outer_params, defaults={"lr": adamw["lr"]})

        self.adamw = torch.optim.AdamW(adamw_params, **adamw)
        self.sgd = torch.optim.SGD(sgd_params, **sgd)

        if not (
            len(self.param_groups)
            == len(self.adamw.param_groups)
            == len(self.sgd.param_groups)
        ):
            raise ValueError("Outer/AdamW/SGD param_groups do not match.")

        self.switch_step = switch_step
        self.transfer_momentum = transfer_momentum
        self.global_step = 0
        self.switched = False

        for outer_g, adam_g, sgd_g in zip(
            self.param_groups, self.adamw.param_groups, self.sgd.param_groups
        ):
            adam_lr = _as_scalar(adam_g["lr"])
            sgd_lr = _as_scalar(sgd_g["lr"])

            if adam_lr == 0 and sgd_lr != 0:
                raise ValueError("adamw lr cannot be 0 when sgd lr is non-zero.")

            outer_g["lr"] = adam_g["lr"]   # scheduler 看到的 base lr
            outer_g["adamw_lr_scale"] = 1.0
            outer_g["sgd_lr_scale"] = 1.0 if adam_lr == 0 else sgd_lr / adam_lr

        self._sync_lrs()

    def _sync_lrs(self):
        for outer_g, adam_g, sgd_g in zip(
            self.param_groups, self.adamw.param_groups, self.sgd.param_groups
        ):
            base_lr = outer_g["lr"]
            adam_g["lr"] = base_lr * outer_g.get("adamw_lr_scale", 1.0)
            sgd_g["lr"] = base_lr * outer_g.get("sgd_lr_scale", 1.0)

    def _maybe_switch(self):
        if self.switched or self.global_step < self.switch_step:
            return

        if self.transfer_momentum:
            for adam_g, sgd_g in zip(self.adamw.param_groups, self.sgd.param_groups):
                for p_adam, p_sgd in zip(adam_g["params"], sgd_g["params"]):
                    st = self.adamw.state.get(p_adam, None)
                    if st is not None and "exp_avg" in st:
                        self.sgd.state[p_sgd]["momentum_buffer"] = (
                            st["exp_avg"].detach().clone()
                        )

        self.switched = True

    def step(self, closure=None):
        self._maybe_switch()
        self._sync_lrs()
        loss = (self.sgd if self.switched else self.adamw).step(closure)
        self.global_step += 1
        return loss

    def zero_grad(self, set_to_none: bool = True):
        # 两个内部 optimizer 指向的是同一组参数，调一个其实也够；
        # 这里两个都调，行为最直观。
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.sgd.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        outer = super().state_dict()
        return {
            "state": outer["state"],
            "param_groups": outer["param_groups"],
            "global_step": self.global_step,
            "switched": self.switched,
            "switch_step": self.switch_step,
            "transfer_momentum": self.transfer_momentum,
            "adamw": self.adamw.state_dict(),
            "sgd": self.sgd.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict.get("global_step", 0)
        self.switched = state_dict.get("switched", False)
        self.switch_step = state_dict.get("switch_step", self.switch_step)
        self.transfer_momentum = state_dict.get(
            "transfer_momentum", self.transfer_momentum
        )

        super().load_state_dict(
            {
                "state": state_dict["state"],
                "param_groups": state_dict["param_groups"],
            }
        )
        self.adamw.load_state_dict(state_dict["adamw"])
        self.sgd.load_state_dict(state_dict["sgd"])
        self._sync_lrs()