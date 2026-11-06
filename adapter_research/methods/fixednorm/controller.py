import torch
from ..isr.controller import SRController
from ..isr.config import SRConfig
from .projection import project_factors


class FixedNormController(SRController):
    def __init__(self, model, config, seed=0):
        super().__init__(model, config, seed)
        if config.frobenius_target > 0 and any(
                factor.A.dtype != torch.float32 or factor.B.dtype != torch.float32
                for factor in self.factors):
            raise ValueError('Fixed-norm training requires FP32 A/B; AMP forward is supported')

    def after_optimizer(self, skipped=False):
        if self.config.frobenius_target > 0:
            self.metrics.update({'projection/target': self.config.frobenius_target,
                                 'projection/optimizer_skipped': int(skipped)})
            if not skipped:
                self.metrics.update(project_factors(self.factors, self.config.frobenius_target,
                                                    self.config.min_frobenius))
        super().after_optimizer(skipped)


def create_controller(model, parameters, seed=0):
    return FixedNormController(model, SRConfig(**parameters), seed)
