import argparse
import uuid
import os
import json
from MergeModels.ActivationMerging import activation_merge

argparser = argparse.ArgumentParser()
argparser.add_argument("--method", type=str, default="ties", help="Method for merging (MergeKit methods). Default: ties")
argparser.add_argument("--base_model", type=str, default="unsloth/llama-2-13b", help="Base model name or path. Default: meta-llama/Llama-3.1-8B-Instruct")
argparser.add_argument("--models_to_merge", type=str, default="WizardLMTeam/WizardLM-13B-V1.2,vanillaOVO/WizardMath-13B-V1.0,layoric/llama-2-13b-code-alpaca", help="Models to merge. Default: WizardLMTeam/WizardLM-13B-V1.2,vanillaOVO/WizardMath-13B-V1.0,layoric/llama-2-13b-code-alpaca")
argparser.add_argument("--save_path", type=str, default="TIES_InstructMathCode", help="Save path for the merged model. Default: TIES_InstructMathCode")

args = argparser.parse_args()


merge_method = args.method
base_model = args.base_model
models_to_merge = args.models_to_merge.split(",")

merger = LMMerger(merge_method = merge_method, base_model = base_model, models_to_merge = combinations[i], save_path = args.save_path)
merger()