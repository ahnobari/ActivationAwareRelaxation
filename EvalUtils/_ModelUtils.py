from vllm import LLM
import torch
import os

def load_model(model_name_or_path, **kwargs):
    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false"
    )
    if 'tensor_parallel_size' not in kwargs:
        kwargs['tensor_parallel_size'] = 1
    if 'trust_remote_code' not in kwargs:
        kwargs['trust_remote_code'] = True
    if 'max_model_len' not in kwargs:
        kwargs['max_model_len'] = 4096
    
    model = LLM(model=model_name_or_path, **kwargs)
    
    return model