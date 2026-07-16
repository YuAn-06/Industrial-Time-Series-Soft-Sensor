"""Central registry and loader for model packages."""

import importlib


MODEL_REGISTRY = {
    "Nystromformer": "models.Nystromformer",
    "DAGRU": "models.DAGRU",
    "DMVAER": "models.DMVAER",
    "VRNN": "models.VRNN",
    "TCVAE": "models.TCVAE",
    "iTransformer": "models.iTransformer",
    "Transformer": "models.Transformer",
    "EnvFormer": "models.EnvFormer",
    "Fredformer": "models.Fredformer",
    "HSAM_dGRUs": "models.HSAM_dGRUs",
    "PatchTST": "models.PatchTST",
    "Autoformer": "models.Autoformer",
    "DLinear": "models.DLinear",
    "ARDNN": "models.ARDNN",
    "MSACNN": "models.MSACNN",
    "CVAESMC": "models.CVAESMC",
    "LDCNN": "models.LDCNN",
    "Nonstationary_Transformer": "models.Nonstationary_Transformer",
    "DMRIFormer": "models.DMRIFormer",
    "Informer": "models.Informer",
    "LSTM": "models.LSTM",
    "VALSTM": "models.VALSTM",
    "Crossformer": "models.Crossformer",
    "TimeMixer": "models.TimeMixer",
    "TimesNet": "models.TimesNet",
    "GTFTS": "models.GTFTS",
    "SparseTSF": "models.SparseTSF",
    "TCN": "models.TCN",
    "TimeFilter": "models.TimeFilter",
    "STALSTM": "models.STALSTM",
    "Koopa": "models.Koopa",
    "TimeKAN": "models.TimeKAN",
    "MSGNet": "models.MSGNet",
    "DLSTM": "models.DLSTM",
    "GCT": "models.GCT",
    "SOFTS": "models.SOFTS",
    "FEDformer": "models.FEDformer",
    "STDTAEm": "models.STDTAEm",
    "GraphSAGE_IMATCN": "models.GraphSAGE_IMATCN",
    "FASConvAELSTM": "models.FASConvAELSTM",
    "TSLambdaGRU": "models.TSLambdaGRU",
}

MODEL_ALIASES = {
    "Envformer": "EnvFormer",
    "Nystroformer": "Nystromformer",
}


def canonical_model_name(model_name):
    """Return the registered spelling for a model name or alias."""
    return MODEL_ALIASES.get(model_name, model_name)


def get_model_package(model_name):
    """Import and return a registered model package."""
    canonical_name = canonical_model_name(model_name)
    try:
        module_path = MODEL_REGISTRY[canonical_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models: {supported}"
        ) from exc

    package = importlib.import_module(module_path)
    for attribute in ("Model", "MODEL_CONFIG", "MODEL_SPEC"):
        if not hasattr(package, attribute):
            raise AttributeError(
                f"Model package '{module_path}' must export '{attribute}'."
            )
    return package


def get_model_spec(model_name):
    return get_model_package(model_name).MODEL_SPEC


def get_model_config_class(model_name):
    return get_model_package(model_name).MODEL_CONFIG
