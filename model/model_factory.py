from model.cluspro_baseline import ClusProBaseline
from model.lhp_czsl import LHPCZSL

def get_model(config, attributes, classes, offset):
    if config.model_name == 'cluspro_baseline':
        model = ClusProBaseline(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'lhp_czsl':
        model = LHPCZSL(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(f"Unknown model: {config.model_name}")
    return model
