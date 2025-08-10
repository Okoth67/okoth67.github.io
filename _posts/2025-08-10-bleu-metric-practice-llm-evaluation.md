# My First BLEU Metric Practice for LLM Evaluation

*August 10, 2025*

As part of my journey into **large language model (LLM) evaluation**, I've been experimenting with well-known metrics used in natural language processing. After recently trying out ROUGE-L, I decided to explore another classic: **BLEU** (Bilingual Evaluation Understudy).

Link to the paper: https://aclanthology.org/P02-1040.pdf

Link to my notebook repo: https://github.com/Okoth67/wekeza_llm_eval_bleu/tree/main
## The BLEU Background

BLEU was introduced in the influential paper *"BLEU: a Method for Automatic Evaluation of Machine Translation"* by Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu at IBM Research. The authors proposed BLEU as a fast, inexpensive, and language-independent alternative to human evaluation, designed to measure how close machine-generated text is to high-quality human references.

Although originally created for machine translation, BLEU has since been widely adopted in other NLP tasks, including summarization and now LLM output evaluation. The metric works by comparing n-gram overlap between predictions and references, with a brevity penalty to discourage overly short outputs.

## My Practical Implementation

For my practice, I used my fine-tuned **Wekeza** model to answer a small set of finance-related questions, then compared its outputs against my own reference answers using BLEU. Here's the setup code I used:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import evaluate

# Load the fine-tuned model
model_path = "./distilgpt2-wekeza-finetuned_v5_cot_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.eval()

# Create test dataset
data = [
    {
        "question": "What is a money market fund in Kenya?",
        "reference": "A money market fund in Kenya is a low-risk collective investment scheme that invests in short-term debt instruments and offers high liquidity."
    },
    {
        "question": "Explain fixed deposits in Kenyan banks.",
        "reference": "Fixed deposits are bank accounts where money is locked for a set period in exchange for a higher interest rate than regular savings accounts."
    }
]

dataset = Dataset.from_list(data)
bleu = evaluate.load("bleu")
```

The generation and evaluation process was straightforward:

```python
predictions = []
references = []

for example in dataset:
    prompt = example["question"]
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    predictions.append(generated_text)
    references.append([example["reference"]])  # BLEU expects list of references

# Compute BLEU score
results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU score: {results['bleu']:.4f}")
```

## Results and Learning

The BLEU score I obtained was **0.0223** which was unsurprisingly low, since my test set was tiny and my prompts were open-ended. But the point of this exercise wasn't to get a high score,  it was to **understand how BLEU works** in practice, how to structure predictions and references, and how to interpret the results.

Key takeaways from this experiment:

* **Reference format matters**: BLEU expects references as lists of strings, even for single references
* **N-gram precision**: BLEU focuses on exact word matches, which can be harsh for creative or paraphrased responses
* **Brevity penalty**: The metric penalizes outputs that are significantly shorter than references
* **Context sensitivity**: Open-ended questions naturally score lower than constrained tasks

## Technical Observations

During implementation, I noticed a few important details:

* The **LoRA adapters** loaded properly despite some warnings about `fan_in_fan_out` settings
* **CPU generation** worked fine for this small-scale evaluation
* **Tokenizer configuration** automatically set `pad_token_id` to `eos_token_id` for generation
* The **Hugging Face evaluate** library made BLEU computation straightforward

## Why This Matters

This notebook is part of my **introductory practice series** on LLM evaluation. I'm keeping these exercises small, focused, and practical so I can build a strong foundation before moving on to more advanced metrics and larger datasets.

BLEU taught me that traditional n-gram metrics have limitations when evaluating conversational or explanatory text. While useful for translation tasks where there's often a "correct" output, BLEU can be overly strict for creative text generation where multiple valid responses exist.

## Next Steps

Moving forward, I plan to explore **METEOR** or **BERTScore** to see how they differ from BLEU in capturing semantic similarity. METEOR incorporates stemming and synonyms, while BERTScore uses contextual embeddings, both potentially more suitable for evaluating LLM outputs than pure n-gram overlap.

I'm also considering:
* Expanding the test dataset for more reliable metrics
* Comparing BLEU scores across different model checkpoints
* Testing how different generation parameters affect BLEU performance

## Resources

This experiment builds on the foundational work by Papineni et al. in their original BLEU paper. While BLEU has known limitations for modern NLP tasks, understanding it remains important for anyone working in text evaluation.

The complete code for this experiment will be available in my GitHub repository once I organize the evaluation notebooks into a coherent series.

---
