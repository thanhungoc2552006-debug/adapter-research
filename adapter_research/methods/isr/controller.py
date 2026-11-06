"""Inverse stable rank of LoRA factors. Never changes adapter parameters."""
from dataclasses import dataclass
import hashlib
import time
import torch
from .config import SRConfig

TARGETS = {'q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj'}


def gram_frobenius_squared(A, B, gamma=1.0):
    """Exact algebraic expression; retain gradients through both Gram matrices."""
    return gamma**2 * ((B.T @ B) * (A @ A.T).T).sum()


def fixed_vector_penalty(A, B, v, gamma=1.0):
    """Keep native dtype for FP64 gradcheck; caller controls FP32 training context."""
    q = gamma * (B @ (A @ v.detach()))
    numerator = q.square().sum()
    denominator = gram_frobenius_squared(A, B, gamma)
    return numerator / denominator, numerator, denominator


@torch.no_grad()
def exact_factor_spectrum(A, B, gamma=1.0):
    """Diagnostic only: QR + SVD on an r-by-r core, not on the full update."""
    _, rb = torch.linalg.qr(B, mode='reduced')
    _, ra = torch.linalg.qr(A.T, mode='reduced')
    squares = torch.linalg.svdvals(gamma * (rb @ ra.T)).square()
    d = squares.sum()
    top = squares.max()
    return d, top


@torch.no_grad()
def power_iteration(A, B, v, iterations):
    """Return normalized right vector or None when the current vector degenerates."""
    tiny = torch.finfo(A.dtype).tiny
    for _ in range(iterations):
        u = B @ (A @ v)
        unorm = torch.linalg.vector_norm(u)
        if not torch.isfinite(unorm).item() or unorm.item() <= tiny:
            return None
        u = u / unorm
        v = A.T @ (B.T @ u)
        vnorm = torch.linalg.vector_norm(v)
        if not torch.isfinite(vnorm).item() or vnorm.item() <= tiny:
            return None
        v = v / vnorm
    return v.detach()


@dataclass
class Factors:
    name: str
    A: torch.Tensor
    B: torch.Tensor
    gamma: float


def select_lora_factors(model, expected_modules=160):
    selected = []
    for name, module in model.named_modules():
        if name.rsplit('.', 1)[-1] not in TARGETS:
            continue
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue
        active = list(module.active_adapters)
        if len(active) != 1:
            raise ValueError(f'ISR requires exactly one active LoRA adapter: {name}')
        adapter = active[0]
        if getattr(module, 'use_dora', {}).get(adapter, False):
            raise ValueError('ISR operates on standard LoRA, not the DoRA update')
        if hasattr(module, 'lora_S') and adapter in module.lora_S:
            raise ValueError('ISR operates on standard LoRA, not PoLAR')
        A, B = module.lora_A[adapter].weight, module.lora_B[adapter].weight
        if not A.requires_grad or not B.requires_grad:
            raise ValueError(f'Selected adapter factors are frozen: {name}')
        gamma = float(module.scaling[adapter])
        if not gamma > 0:
            raise ValueError(f'Positive LoRA scaling required: {name}')
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            raise ValueError(f'Unexpected factor shapes: {name}')
        selected.append(Factors(name, A, B, gamma))
    selected.sort(key=lambda factor: factor.name)
    if not selected or (expected_modules and len(selected) != expected_modules):
        raise ValueError(f'Expected {expected_modules or "some"} LoRA modules, found {len(selected)}')
    return selected


class SRController:
    def __init__(self, model, config=SRConfig(), seed=0):
        self.config = config
        self.seed = int(seed)
        self.factors = select_lora_factors(model, config.expected_modules)
        self.total_updates = None
        self.successful_updates = 0
        self.attempted_updates = 0
        self.vectors = {}
        self.generators = {}
        self.metrics = {}
        self.aux_backward_calls = 0

    def configure(self, total_updates):
        total_updates = int(total_updates)
        if self.total_updates is not None and self.total_updates != total_updates:
            raise ValueError('Resume changed Trainer max_steps; use a new experiment')
        self.total_updates = total_updates

    def coefficient(self):
        return self.config.schedule(self.successful_updates, self.total_updates)

    def _generator(self, name):
        if name not in self.generators:
            digest = hashlib.sha256(f'{self.seed}|{name}'.encode()).digest()
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int.from_bytes(digest[:8], 'big') % (2**63 - 1))
            self.generators[name] = generator
        return self.generators[name]

    def _random_vector(self, factor, A):
        v = torch.randn(A.shape[1], generator=self._generator(factor.name), dtype=A.dtype)
        v = v.to(A.device)
        return v / torch.linalg.vector_norm(v)

    def _vector(self, factor, A, B):
        cached = self.vectors.get(factor.name)
        for attempt in range(self.config.max_restarts + 1):
            fresh = cached is None or attempt > 0
            v = self._random_vector(factor, A) if fresh else cached.to(A.device, A.dtype)
            iterations = self.config.pi_init_iters if fresh else self.config.pi_iters
            result = power_iteration(A, B, v, iterations)
            if result is not None:
                self.vectors[factor.name] = result.detach().clone()
                return result, attempt
        self.vectors.pop(factor.name, None)
        return None, self.config.max_restarts

    def weighted_loss(self):
        """Called once after CE accumulation, while all gradients are still scaled."""
        coefficient = self.coefficient()
        self.metrics = {'sr/lambda': coefficient, 'sr/modules_total': len(self.factors),
                        'sr/modules_active': 0, 'sr/skipped_small_norm': 0,
                        'sr/pi_failures': 0, 'sr/pi_restarts': 0, 'sr/aux_raw': 0.,
                        'sr/aux_weighted': 0.}
        if coefficient == 0:
            return None  # no Gram/PI/vector/RNG work during warm-up or lambda=0
        started = time.perf_counter()
        terms, norms, tops, ratios, exact_sr = [], [], [], [], []
        # First active update and periodic audits cover every regularized module.
        audit = not self.vectors or (self.attempted_updates + 1) % self.config.audit_every == 0
        for factor in self.factors:
            with torch.autocast(device_type=factor.A.device.type, enabled=False):
                A, B = factor.A.float(), factor.B.float()
                if not torch.isfinite(A).all().item() or not torch.isfinite(B).all().item():
                    raise FloatingPointError(f'Nonfinite LoRA parameters: {factor.name}')
                d = gram_frobenius_squared(A, B, factor.gamma)
                value = d.detach().item()
                if not torch.isfinite(d.detach()).item() or value < 0:
                    raise FloatingPointError(f'Invalid Gram Frobenius norm: {factor.name}: {value}')
                if value <= self.config.min_frobenius**2:
                    self.metrics['sr/skipped_small_norm'] += 1
                    continue
                v, resets = self._vector(factor, A.detach(), B.detach())
                self.metrics['sr/pi_restarts'] += resets
                if v is None:
                    self.metrics['sr/pi_failures'] += 1
                    continue
                q = factor.gamma * (B @ (A @ v.detach()))
                s = q.square().sum()
                term = s / d  # never detach denominator
                if not torch.isfinite(term.detach()).item() or term.detach().item() > 1.001:
                    raise FloatingPointError(f'Invalid ISR penalty: {factor.name}')
                terms.append(term)
                norms.append(value**0.5)
                tops.append(s.detach().item()**0.5)
                if audit:
                    exact_d, exact_top = exact_factor_spectrum(A.detach(), B.detach(), factor.gamma)
                    ratios.append((s.detach() / exact_top).item())
                    exact_sr.append((exact_d / exact_top).item())
        self.metrics['sr/modules_active'] = len(terms)
        if not terms:
            self.metrics['sr/compute_seconds'] = time.perf_counter() - started
            return None
        raw = torch.stack(terms).sum() / len(self.factors)  # M is fixed, not active count
        weighted = coefficient * raw
        self.metrics.update({'sr/aux_raw': raw.detach().item(),
                             'sr/aux_weighted': weighted.detach().item(),
                             'sr/frobenius_mean': sum(norms) / len(norms),
                             'sr/pi_spectral_mean': sum(tops) / len(tops)})
        if ratios:
            self.metrics.update({'sr/pi_sigma2_ratio_min': min(ratios),
                                 'sr/pi_sigma2_ratio_mean': sum(ratios) / len(ratios),
                                 'sr/exact_stable_rank_mean_all': sum(exact_sr) / len(exact_sr)})
        self.metrics['sr/compute_seconds'] = time.perf_counter() - started
        return weighted

    def after_optimizer(self, skipped=False):
        self.attempted_updates += 1
        if not skipped:
            self.successful_updates += 1
        self.metrics.update({'sr/successful_updates': self.successful_updates,
                             'sr/attempted_updates': self.attempted_updates,
                             'sr/optimizer_skipped': int(skipped),
                             'sr/aux_backward_calls': self.aux_backward_calls})

    def state_dict(self):
        return {'version': self.config.version, 'config_digest': self.config.digest(),
                'seed': self.seed, 'total_updates': self.total_updates,
                'successful_updates': self.successful_updates, 'attempted_updates': self.attempted_updates,
                'aux_backward_calls': self.aux_backward_calls,
                'shapes': {f.name: [list(f.A.shape), list(f.B.shape), f.gamma] for f in self.factors},
                'vectors': {name: v.detach().cpu().clone() for name, v in self.vectors.items()},
                'generators': {name: g.get_state().clone() for name, g in self.generators.items()}}

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ['version', 'config_digest', 'seed', 'shapes']:
            if state[key] != expected[key]:
                raise ValueError(f'ISR checkpoint mismatch: {key}')
        names = {f.name: f for f in self.factors}
        if not set(state['vectors']) <= set(names) or not set(state['generators']) <= set(names):
            raise ValueError('Unexpected module in ISR checkpoint')
        self.total_updates = state['total_updates']
        self.successful_updates = int(state['successful_updates'])
        self.attempted_updates = int(state['attempted_updates'])
        self.aux_backward_calls = int(state['aux_backward_calls'])
        if not 0 <= self.successful_updates <= self.attempted_updates:
            raise ValueError('Invalid ISR optimizer counters')
        self.vectors = {}
        for name, v in state['vectors'].items():
            if (list(v.shape) != [names[name].A.shape[1]] or not torch.isfinite(v).all().item()
                    or abs(v.float().norm().item()-1.) > 1e-3):
                raise ValueError(f'Invalid cached PI vector: {name}')
            self.vectors[name] = v.to(names[name].A.device, dtype=torch.float32)
        self.generators = {}
        for name, generator_state in state['generators'].items():
            self._generator(name).set_state(generator_state.cpu())


def create_controller(model, parameters, seed=0):
    return SRController(model, SRConfig(**parameters), seed)
