from model.cluspro_baseline import ClusProBaseline

def get_model(config, attributes, classes, offset):
    if config.model_name == 'cluspro_baseline':
        model = ClusProBaseline(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(f"Unknown model: {config.model_name}")
    return model
