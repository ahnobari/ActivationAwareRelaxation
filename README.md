# ActivationAwareRelaxation
![image](https://github.com/user-attachments/assets/516effa0-a981-4422-a09d-ac2cac40c16e)
Model merging, a technique that combines parameters and embeddings from multiple fine-tuned models, has emerged as a promising strategy to enhance large language model (LLM) performance across diverse tasks while maintaining computational efficiency. In this study, we explore the potential benefits of looking at the activation space of large language models to improve the performance and robustness of different merging methods. To incorporate the activation space in model merging, we proposed activation-aware relaxation (AWR) as a second-layer solution that can be applied to any merging method. AWR aims to ensure robustness by maintaining the salient weights of the base model while allowing the merging methods to leverage the expertise of fine-tuned models. AWR takes inspiration from principles of continual learning and model compression and leverages a task-agnostic calibration set to prioritize critical weights during the merging process. We evaluate the performance benefits of the proposed technique on various merging methods in different scenarios and show that AWR improves the performance of merged language models on numerous benchmarks. Our work shows that the activation space encompasses vital information that can reveal potentially transformative avenues for further improvements in merging large language models.

# Usage
For merging methods other than WIDEN you can run the following command (example for TIES merging):

```bash
python merge.py --method ties --base_model unsloth/llama-2-13b --models_to_merge WizardLMTeam/WizardLM-13B-V1.2,vanillaOVO/WizardMath-13B-V1.0,layoric/llama-2-13b-code-alpaca --save_path ./TIES_InstructMathCode
```

To apply AWR to a given checkpoint say the merge results from the example above you can run:

```bash
python performAWR.py --merged_model ./TIES_InstructMathCode --pretrained_model_name unsloth/llama-2-13b --omega 0.4 --save_path ./TIES_AWR_InstructMathCode
```

To run benchamraks on a given model you can run (example of running the AWR from above):

```bash
python evaluate_model.py --model ./TIES_AWR_InstructMathCode
```
