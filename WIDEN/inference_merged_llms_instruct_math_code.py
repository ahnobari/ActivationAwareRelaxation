import argparse
import sys
import logging
import os
import time
from vllm import LLM

from utils.evaluate_llms_utils import test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for inference of merged LLMs")
    parser.add_argument("--checkpoint_path", type=str, help="path to the checkpoint file")
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
    parser.add_argument("--evaluate_source_model_name", type=str, required=True, help="evaluate source model name, used for loading tokenizer")
    parser.add_argument("--evaluate_task", type=str, help="evaluate task", default="instruct", choices=["instruct", "math", "code"])
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    logger.info(f"Configuration is {args}.")

    load_model_path = args.checkpoint_path
    tok_path = os.path.join(load_model_path, args.evaluate_source_model_name) if args.evaluate_source_model_name!='' else load_model_path
    llm = LLM(model=load_model_path, tokenizer=tok_path, tensor_parallel_size=args.tensor_parallel_size)

    if args.evaluate_task == "instruct":
        logger.info(f"Evaluating merged model on instruct task...")
        save_gen_results_folder = f"./save_gen_instruct_responses_results/{load_model_path.split('/')[-1]}"
        test_alpaca_eval(llm=llm, generator_model_name=load_model_path, logger=logger, start_index=args.start_index, end_index=args.end_index,
                         save_gen_results_folder=save_gen_results_folder)
    elif args.evaluate_task == "math":
        logger.info(f"Evaluating merged model on math task...")
        save_gen_results_folder = f"./save_gen_mathematics_results/{load_model_path.split('/')[-1]}"
        test_data_path = "math_code_data/gsm8k_test.jsonl"
        test_gsm8k(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                   start_index=args.start_index, end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
        save_gen_results_folder = f"./save_gen_mathematics_results/{load_model_path.split('/')[-1]}"
        test_data_path = "math_code_data/MATH_test.jsonl"
        test_hendrycks_math(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                            start_index=args.start_index, end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    else:
        assert args.evaluate_task == "code"
        logger.info(f"Evaluating merged model on code task...")
        save_gen_results_folder = f"./save_gen_codes_results/human_eval/{load_model_path.split('/')[-1]}"
        test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                        save_gen_results_folder=save_gen_results_folder)
        save_gen_results_folder = f"./save_gen_codes_results/mbpp/{load_model_path.split('/')[-1]}"
        test_data_path = "math_code_data/mbpp.test.jsonl"
        test_mbpp(llm=llm, test_data_path=test_data_path, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                  save_gen_results_folder=save_gen_results_folder)

    logger.info(f"Inference of merged model {load_model_path} on {args.evaluate_task} task is completed.")

    sys.exit()
