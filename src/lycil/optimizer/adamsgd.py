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


class AdamThenSGD(Optimizer):
    """
    - 外部只维护一个统一 lr
    - Adam / SGD 都共用这个 lr
    - scheduler 挂在外层 optimizer 上即可
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        adam: dict,
        sgd: dict,
        switch_step: int,
        transfer_momentum: bool = False,
    ):
        outer_params = _clone_param_spec(params)
        adam_params = _clone_param_spec(params)
        sgd_params = _clone_param_spec(params)

        adam_cfg = dict(adam)
        sgd_cfg = dict(sgd)

        # 不允许在内部配置里再单独传 lr
        if "lr" in adam_cfg or "lr" in sgd_cfg:
            raise ValueError("Do not pass `lr` inside `adam` or `sgd`; use outer `lr`.")

        adam_cfg["lr"] = lr
        sgd_cfg["lr"] = lr

        super().__init__(outer_params, defaults={"lr": lr})

        self.adam = torch.optim.Adam(adam_params, **adam_cfg)
        self.sgd = torch.optim.SGD(sgd_params, **sgd_cfg)

        if not (
            len(self.param_groups)
            == len(self.adam.param_groups)
            == len(self.sgd.param_groups)
        ):
            raise ValueError("Outer/Adam/SGD param_groups do not match.")

        self.switch_step = switch_step
        self.transfer_momentum = transfer_momentum
        self.global_step = 0
        self.switched = False

        self._sync_lrs()

    def _sync_lrs(self):
        for outer_g, adam_g, sgd_g in zip(
            self.param_groups, self.adam.param_groups, self.sgd.param_groups
        ):
            shared_lr = outer_g["lr"]
            adam_g["lr"] = shared_lr
            sgd_g["lr"] = shared_lr

    def _maybe_switch(self):
        if self.switched or self.global_step < self.switch_step:
            return

        if self.transfer_momentum:
            for adam_g, sgd_g in zip(self.adam.param_groups, self.sgd.param_groups):
                for p_adam, p_sgd in zip(adam_g["params"], sgd_g["params"]):
                    st = self.adam.state.get(p_adam, None)
                    if st is not None and "exp_avg" in st:
                        self.sgd.state[p_sgd]["momentum_buffer"] = (
                            st["exp_avg"].detach().clone()
                        )

        self.switched = True

    def step(self, closure=None):
        self._maybe_switch()
        self._sync_lrs()
        loss = (self.sgd if self.switched else self.adam).step(closure)
        self.global_step += 1
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.adam.zero_grad(set_to_none=set_to_none)
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
            "adam": self.adam.state_dict(),
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
        self.adam.load_state_dict(state_dict["adam"])
        self.sgd.load_state_dict(state_dict["sgd"])
        self._sync_lrs()