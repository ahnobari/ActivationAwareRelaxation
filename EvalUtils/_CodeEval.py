from evalplus.data import get_human_eval_plus, get_mbpp_plus
from tqdm.auto import tqdm
from ._CodeTools import evaluate_minimal
from vllm import SamplingParams
from evalplus.codegen import sanitize

system_chat_prompt = "You are an intelligent programming assistant to produce Python algorithmic solutions"

human_eval_user_chat_template = """user
Can you complete the following Python function?
```python
{prompt}
```
"""

huma_eval_decorator = """Can you complete the following Python function?
```python
{prompt}
"""

human_eval_assistant_chat_template = """```pythonMAGICSPLIT"""

def run_code_test(model, dataset, chat_mode=False):
    if dataset == "humaneval":
        data = get_human_eval_plus()
    elif dataset == "mbpp":
        data = get_mbpp_plus()
    else:
        raise ValueError(f"Invalid dataset {dataset}. Must be humaneval or mbpp.")
    
    prompts = []
    entry_points = []
    task_ids = []
    for task_id, task in tqdm(data.items()):
        # prompt = task['prompt']
        if not chat_mode:
            if dataset == "humaneval":
                 prompts.append(huma_eval_decorator.format(prompt=task['prompt']))
            else:
                prompts.append(task['prompt'])
        elif dataset != "humaneval":
            messages = [
                {
                    "role": "system",
                    "content": system_chat_prompt,
                },
                {
                    "role": "user",
                    "content": task['prompt']
                }
            ]
            messages = model.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts.append(messages)
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_chat_prompt,
                },
                {
                    "role": "user",
                    "content": human_eval_user_chat_template.format(prompt=task['prompt'])
                },
                {
                    "role": "assistant",
                    "content": human_eval_assistant_chat_template
                }
            ]
            messages = model.get_tokenizer().apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            messages = messages.split("MAGICSPLIT")[0]
            prompts.append(messages)
            
        entry_points.append(task['entry_point'])
        task_ids.append(task_id)
    
    sampling = SamplingParams(n=1, temperature=0.0, max_tokens=1024, top_p=1.0)
    
    results = model.generate(prompts, sampling_params=sampling)
    results = extract_outputs(results)
    
    sanitized_results = []

    for result, entry_point in tqdm(zip(results, entry_points), total=len(results)):
        sanitized_result = sanitize(result, entry_point)
        sanitized_results.append(sanitized_result)
        
    samples = []
    for task_id, result in zip(task_ids, sanitized_results):
        samples.append({"task_id": task_id, "solution": result})
        
    final_acc = evaluate_minimal(dataset, samples)
    
    return final_acc

def extract_outputs(outputs):
    return [output.outputs[0].text.replace("\t", "    ") for output in outputs]