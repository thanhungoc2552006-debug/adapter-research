"""Serializable ISR configuration; this module does not import PyTorch."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class SRConfig:
    lambda_max: float = 0.01
    warmup_ratio: float = 0.05
    ramp_ratio: float = 0.10
    min_frobenius: float = 1e-6
    pi_init_iters: int = 20
    pi_iters: int = 5
    max_restarts: int = 3
    audit_every: int = 20
    expected_modules: int = 160
    frobenius_target: float = 0.0
    version: str = 'isr-v2'

    def __post_init__(self):
        for name in ['lambda_max', 'warmup_ratio', 'ramp_ratio', 'min_frobenius', 'frobenius_target']:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if 0 < self.frobenius_target <= self.min_frobenius:
            raise ValueError('frobenius_target must exceed min_frobenius, or be 0 to disable projection')
        if self.warmup_ratio >= 1 or self.ramp_ratio > 1:
            raise ValueError('warmup_ratio must be <1; ramp_ratio must be <=1')
        for name in ['pi_init_iters', 'pi_iters', 'audit_every']:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ['max_restarts', 'expected_modules']:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')

    def to_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def schedule(self, successful_updates, total_updates):
        if total_updates < 1:
            raise ValueError('Trainer must configure total_updates before ISR is used')
        warmup = math.ceil(self.warmup_ratio * total_updates)
        ramp = max(1, math.ceil(self.ramp_ratio * total_updates))
        if successful_updates < warmup:
            return 0.0
        return self.lambda_max * min(1.0, (successful_updates - warmup + 1) / ramp)
