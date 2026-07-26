
DEFAULT_KWARGS = {
    "max_new_tokens": 128,
    "steps": 128,
    "block_length": 64,
}

DATASET_CONFIGS = {
    "M3CoT": {
        "max_new_tokens": 512,
        "steps": 256,
        "block_length": 64,
    },
    "M3CoT_COT": {
        "max_new_tokens": 512,
        "steps": 256,
        "block_length": 64,
    },
    "M3CoT_DIRECT": {
        "max_new_tokens": 64,
        "steps": 64,
        "block_length": 32,
    },
    "CHAIR": {
        "max_new_tokens": 128,
        "steps": 128,
        "block_length": 64,
    },
    "MathVista_MINI": {
        "max_new_tokens": 96,
        "steps": 96,
        "block_length": 48,
    },
    
    "MathVerse_MINI_Vision_Only": {
        "max_new_tokens": 256,
        "steps": 128,
        "block_length": 32,
    },
    
    "MMVet": {
        "max_new_tokens": 512,
        "steps": 256,
        "block_length": 128,
    },
    "MMMU_DEV_VAL": {
        "max_new_tokens": 128,
        "steps": 128,
        "block_length": 64,
    },
    "MathVision": {
        "max_new_tokens": 512,
        "steps": 256,
        "block_length": 64,
    },
    "MathVision_MINI": {
        "max_new_tokens": 512,
        "steps": 256,
        "block_length": 64,
    },
    "LLaVABench": {
        "max_new_tokens": 256,
        "steps": 128,
        "block_length": 256,
    },
    "VLind-Bench": {
        "max_new_tokens": 16,
        "steps": 16,
        "block_length": 16,
    },
    "ScienceQA_VAL": {
        "max_new_tokens": 256,
        "steps": 128,
        "block_length": 64,
    },
    "ScienceQA_TEST": {
        "max_new_tokens": 256,
        "steps": 128,
        "block_length": 64,
    },
}


def get_dataset_config(dataset_name):
    return DATASET_CONFIGS.get(dataset_name, {})


def merge_configs(*configs):
    result = DEFAULT_KWARGS.copy()
    for config in configs:
        if config:
            result.update(config)
    return result 