
DEFAULT_KWARGS = {
    "max_new_tokens": 128,
    "steps": 128,
    "block_length": 64,
}

DATASET_CONFIGS = {
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
    "MMBench_DEV_EN_2C": {
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
    # M3CoT is NOT registered in upstream Gen-Verse/open-compass VLMEvalKit.
    # Generation uses this schedule via generate_mmada(..., dataset='M3CoT');
    # scoring uses LightChen233/M3CoT evaluate.py (see VLind-Bench/eval/mmada_m3cot_eval.py).
    "M3CoT": {
        "max_new_tokens": 512,
        "steps": 256,
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