from dataclasses import replace
from ..isr import SPEC as ISR

SPEC = replace(ISR, id='isr_fixednorm', label='ISR fixed F', aliases=('lora_isr_fixednorm',),
    defaults={**ISR.defaults, 'lambda_max': .1, 'frobenius_target': 1., 'version': 'isr-fixednorm-v2'},
    controller_factory='adapter_research.methods.fixednorm.controller:create_controller',
    source_modules=ISR.source_modules + (__name__, 'adapter_research.methods.fixednorm.controller', 'adapter_research.methods.fixednorm.projection'),
    label_parameters=(('lambda_max', 'lambda'), ('frobenius_target', 'rho')))
