"""Post-AdamW radial rescaling of each effective LoRA update; no dense BA."""
import math
import re
import time
import torch


@torch.no_grad()
def core_spectrum(A, B, gamma):
    """FP64 QR core handles cancellation without a Gram trace near zero."""
    with torch.autocast(device_type=A.device.type, enabled=False):
        rb = torch.linalg.qr(B.double(), mode='reduced').R
        ra = torch.linalg.qr(A.double().T, mode='reduced').R
        core = float(gamma) * (rb @ ra.T)
        singular = torch.linalg.svdvals(core)
        if not torch.isfinite(singular).all().item():
            raise FloatingPointError('Nonfinite adapter spectrum during projection')
        energy = singular.square().sum().item()
        top = singular.max().item()
        return math.sqrt(energy), top, energy / top**2 if top > 0 else None


@torch.no_grad()
def project_factors(factors, rho, min_frobenius=1e-6, tolerance=2e-5):
    """B <- rho/||gamma BA||_F B; A and AdamW moments remain unchanged.

    Preflight all factors before mutation. Near-zero updates remain unchanged
    and are counted, since zero cannot be scaled to positive norm.
    """
    if not math.isfinite(rho) or rho <= min_frobenius or min_frobenius < 0:
        raise ValueError('rho must be finite and > min_frobenius >= 0')
    started = time.perf_counter()
    plans = []
    for factor in factors:
        if factor.A.dtype not in (torch.float32, torch.float64) or factor.B.dtype != factor.A.dtype:
            raise ValueError('Fixed-norm projection requires FP32/FP64 adapter parameters (AMP forward is supported)')
        if not math.isfinite(factor.gamma) or factor.gamma <= 0:
            raise ValueError('Projection requires finite positive gamma')
        if not torch.isfinite(factor.A).all().item() or not torch.isfinite(factor.B).all().item():
            raise FloatingPointError(f'Nonfinite factors before projection: {factor.name}')
        before, _, _ = core_spectrum(factor.A, factor.B, factor.gamma)
        if before <= min_frobenius:
            plans.append((factor, before, None, None))
            continue
        scale = rho / before
        candidate = (factor.B.double() * scale).to(factor.B.dtype)
        if not torch.isfinite(candidate).all().item():
            raise FloatingPointError(f'Projection overflows B: {factor.name}')
        after, spectral, stable_rank = core_spectrum(factor.A, candidate, factor.gamma)
        if abs(after / rho - 1) > tolerance:
            raise FloatingPointError(f'Projection norm check failed: {factor.name}: {after} vs {rho}')
        plans.append((factor, before, candidate, (scale, after, spectral, stable_rank)))

    metrics = {'projection/target': rho, 'projection/modules_total': len(plans),
               'projection/modules_projected': 0, 'projection/skipped_small_norm': 0}
    pre, post, spectral_post, sr_post, scales, errors = [], [], [], [], [], []
    for factor, before, candidate, measured in plans:
        pre.append(before)
        if candidate is None:
            metrics['projection/skipped_small_norm'] += 1
            after, spectral, stable_rank = core_spectrum(factor.A, factor.B, factor.gamma)
        else:
            factor.B.copy_(candidate)  # preserves Parameter identity and optimizer state
            scale, after, spectral, stable_rank = measured
            scales.append(scale)
            errors.append(abs(after / rho - 1))
            metrics['projection/modules_projected'] += 1
        post.append(after)
        spectral_post.append(spectral)
        if stable_rank is not None:
            sr_post.append(stable_rank)
        match = re.search(r'layers\.(\d+)\.(.+)', factor.name)
        key = f'layer{match.group(1)}.{match.group(2)}' if match else factor.name
        metrics[f'projection/{key}/frobenius_pre'] = before
        metrics[f'projection/{key}/frobenius_post'] = after
        metrics[f'projection/{key}/spectral_post'] = spectral
        if stable_rank is not None:
            metrics[f'projection/{key}/stable_rank_post'] = stable_rank
    if post:
        metrics.update({'projection/frobenius_pre_mean': sum(pre)/len(pre),
                        'projection/frobenius_post_mean': sum(post)/len(post),
                        'projection/frobenius_post_min': min(post),
                        'projection/frobenius_post_max': max(post),
                        'projection/spectral_post_mean': sum(spectral_post)/len(post),
                        'projection/target_relative_error_max': max(errors, default=0.)})
    if sr_post:
        metrics['projection/stable_rank_post_mean'] = sum(sr_post)/len(sr_post)
    if scales:
        metrics.update({'projection/scale_min': min(scales), 'projection/scale_max': max(scales)})
    metrics['projection/compute_seconds'] = time.perf_counter() - started
    return metrics
