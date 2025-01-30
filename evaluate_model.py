import argparse
from EvalUtils import *
import uuid
import os
import json
from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name or path. Default: meta-llama/Llama-3.2-1B-Instruct")
argparser.add_argument("--benchmarks", type=str, default="mmlu,ifeval,mbpp,humaneval", help="Non-Math benchmarks to run. Default: mmlu,ifeval,mbpp,humaneval")
argparser.add_argument("--n_gpu", type=int, default=None, help="Number of GPUs to use. Default: all available")
argparser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for model. Default: bfloat16")
argparser.add_argument("--max_model_len", type=int, default=4096, help="Max model length. Default: 4096")
argparser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template to benchmarks. Default: False")
argparser.add_argument("--batch_size", type=str, default='auto', help="Batch size for model. Default: auto")

argparser.add_argument("--math_benchmarks", type=str, default="math,gsm8k", help="Math benchmarks to run. Default: gsm8k,math")
argparser.add_argument("--data_dir", type=str, default="Data", help="Data directory. Default: Data")
argparser.add_argument("--math_prompt_type", type=str, default="wcot", help="Prompt type for Math benchmarks. Default: wcot")
argparser.add_argument("--math_max_tokens", type=int, default=1024, help="Max tokens for Math benchmarks. Default: 1024")

argparser.add_argument("--mmlu_pro", action="store_true", help="Also run MMLU-Pro. Default: False")

argparser.add_argument("--save_path", type=str, default="results", help="Save path for results. Default: results, File name will be automatically generated")

args = argparser.parse_args()

# check if results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

print("loading model ...")

if args.batch_size != 'auto':
    batch_size = int(args.batch_size)
else:
    batch_size = 'auto'

if args.n_gpu is None:
    args.n_gpu = torch.cuda.device_count()    

# Load model
harness_model = VLLM(
    pretrained=args.model,
    dtype=args.dtype,
    max_model_len=args.max_model_len,
    batch_size=batch_size,
    tensor_parallel_size=args.n_gpu
)

print("running lm harness benchmarks ...")
# Run Benchmarks
results = simple_evaluate(harness_model,
                          tasks=args.benchmarks.split(","),
                          apply_chat_template=args.apply_chat_template,
                          confirm_run_unsafe_code=True)['results']

# Run MMLU Pro If Needed
if args.mmlu_pro:
    print("running MMLU-Pro benchmark ...")
    mmlu_pro_results = run_mmlu_pro(model.model)
    results['mmlu_pro'] = mmlu_pro_results

# Run Math Benchmarks
if args.math_benchmarks == "":
    math_benchmarks = []
else:
    math_benchmarks = args.math_benchmarks.split(",")

for benchmark in math_benchmarks:
    print(f"running {benchmark} benchmark ...")
    math_bench = run_math_benchmark(harness_model.model, benchmark, args.math_prompt_type, data_dir=args.data_dir, max_tokens_per_call=args.math_max_tokens)
    results[benchmark] = math_bench

# save results
print("saving results ...")

while True:
    model_safe = args.model.split("/")[-1].replace("-", "_").replace('.', '_')
    save_path = os.path.join(args.save_path, f"{model_safe}_{uuid.uuid4()}.json")
    if not os.path.exists(save_path):
        break


with open(save_path, "w") as f:
    json.dump(results, f)