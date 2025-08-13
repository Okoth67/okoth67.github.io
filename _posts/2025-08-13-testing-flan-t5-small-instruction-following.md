# Testing Flan-T5 Small for Instruction Following

*August 13, 2025*

I recently explored Flan-T5 small, one of Google Research's publicly released instruction-finetuned models, in a quick evaluation experiment.

## Background

This work is inspired by the paper [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416), which shows how training on thousands of tasks phrased as natural language instructions improves a model's ability to generalize to unseen prompts. The research also highlights how adding chain-of-thought reasoning data enhances logical reasoning skills.

## Simple Test Setup

I loaded the `google/flan-t5-small` model and gave it diverse prompts to test instruction following:

```python
!pip install transformers datasets --quiet

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

prompts = [
    "Explain compound interest in simple terms.",
    "Translate 'How are you today?' into Swahili.",
    "Give me 3 tips for saving money in Kenya.",
    "Summarize: Artificial Intelligence can assist in financial decision making."
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Flan-T5 Response: {response}\n")
```

## Results

Despite its small size, Flan-T5 followed the instructions effectively, producing coherent, context-aware outputs without any additional fine-tuning. The model handled explaining compound interest, translating into Swahili, offering money-saving tips for Kenya, and summarizing a statement on AI in finance.

## Key Takeaway

This small-scale demo reinforces one of the paper's core findings: instruction tuning enables zero-shot generalization across a wide range of tasks. Even the smallest model in the Flan-T5 family can handle multiple task types with surprising accuracy, making it ideal for lightweight, local experimentation.

The complete code is available in my [GitHub repository](https://github.com/Okoth67/llm_eval_flan_t5_instruction/tree/main).

---
