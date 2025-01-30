import argparse
from MergeModels.ActivationMerging import relax_on_merged

argparser = argparse.ArgumentParser()
argparser.add_argument("--merged_model", type=str, help="Path to the merged model.")
argparser.add_argument("--pretrained_model_name", type=str, default= 'unsloth/llama-2-13b', help="Name of the pretrained model to use for relaxation. Default: unsloth/llama-2-13b")
argparser.add_argument("--omega", type=float, default=0.4, help="Value of omega for relaxation. Default: 0.4")
argparser.add_argument("--save_path", type=str, help="Path to save the relaxed model.")

args = argparser.parse_args()

print(f"Relaxing {args.merged_model}...")
merged_model, merged_tokenizer = relax_on_merged(args.merged_model,pretrained_model_name = args.pretrained_model_name, omega=args.omega)
print(f"Saving to {args.save_path}...")
merged_model.save_pretrained(args.save_path)