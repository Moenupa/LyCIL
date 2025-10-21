import torch

from ..data.cil_datamodule import BaseCILDataModule
from .lwf import LWF


class ICaRL(LWF):
    """
    `iCaRL`_: Incremental Classifier and Representation Learning. (Rebuffi et al., CVPR 2017).
    - Exemplar memory: herding + NME-based evaluation

    Args:
        distill_T (float, optional): Temperature for distillation. Default: 2.0.
        lambda_distill (float, optional): Weight for distillation loss. Default: 1.0.
        args: See :class:`BaseIncremental` for other args.
        kwargs: See :class:`BaseIncremental` for other args.

    .. _iCaRL:
        https://arxiv.org/abs/1611.07725
    """

    def __init__(self, *args, mem_size: int = 2000, **kwargs):
        assert mem_size > 0, "iCaRL requires a non-zero memory size."
        super().__init__(*args, mem_size=mem_size, **kwargs)

    @torch.no_grad()
    def update_memory(self, datamodule: BaseCILDataModule, **kwargs) -> None:
        r"""
        Nearest-Mean-of-Exemplars Classification.

        .. math::
        
            \begin{aligned}
            &\textbf{input}: \text{image} \: x \\
            &\textbf{require}: \text{class exemplar sets} \: \mathcal{P}=(P_1,\dots,P_t) \\
            &\textbf{require}: \text{feature map} \: \phi : \mathcal{X} \mapsto \mathbb{R}^d \\
            &\hspace{5mm}\textbf{for} \: y = 1, \dots, t \: \textbf{do} \\
            &\hspace{10mm} \mu_y \gets \frac{1}{|P_y|} \sum_{p \in P_y} \phi(p) \\
            &\hspace{5mm}\textbf{end for} \\
            &\hspace{5mm}y^* = \text{argmin}_y || \phi(x) - \mu_y || \\
            &\textbf{output}: \text{class label} \: y^*
            \end{aligned}
        """
        # reduce exemplar set (for all seen classes)
        num_exemplars_per_class = self.buffer.exemplars_per_class(
            len(datamodule.classes_seen)
        )
        self.buffer.reduce_exemplars(num_exemplars_per_class)

        # construct exemplar set (for current classes)
        for c in self.current_classes:
            self.buffer.build_exemplars_herding(
                model=self,
                feature_extractor=self.feature_extractor,
                dataloader=datamodule.train_dataloader(),
                class_id=c,
                m=num_exemplars_per_class,
                device=self.device,
            )

        # datamodule use `train_dataloader` to access this buffer during next task training
        datamodule.buffer = self.buffer
        return
