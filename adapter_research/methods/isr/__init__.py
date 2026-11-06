from ...registry import MethodSpec

SPEC = MethodSpec('isr', 'ISR', aliases=('lora_isr',),
    defaults={'lambda_max': .01, 'warmup_ratio': .05, 'ramp_ratio': .10,
              'min_frobenius': 1e-6, 'pi_init_iters': 20, 'pi_iters': 5,
              'max_restarts': 3, 'audit_every': 20, 'expected_modules': 160,
              'frobenius_target': 0., 'version': 'isr-v2'},
    validate_parameters='adapter_research.methods.isr.config:SRConfig',
    controller_factory='adapter_research.methods.isr.controller:create_controller',
    source_modules=(__name__, 'adapter_research.methods.isr.controller', 'adapter_research.methods.isr.config'),
    label_parameters=(('lambda_max', 'lambda'),))
