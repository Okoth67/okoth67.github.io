# My First Steps into LLM Evaluation with ROUGE-L

*August 9, 2025*

I've recently started exploring how to evaluate **Large Language Models (LLMs)**, and my first experiment focuses on the **ROUGE-L** metric. This is part of my learning journey, not a polished project, but a hands-on way to understand how to measure an LLM's performance.

## Why ROUGE-L?

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a popular metric for comparing generated text against a reference. The **ROUGE-L** variant looks at the *Longest Common Subsequence* between generated and reference text, making it useful for tasks like summarization and question answering.

In the [original ROUGE paper](https://aclanthology.org/W04-1013.pdf), the authors show how ROUGE-L captures fluency and relevance without requiring exact word matches. This flexibility makes it particularly valuable when evaluating natural language generation tasks.

## What I Did

In my notebook, I:

1. **Loaded my fine-tuned GPT-2 LoRA model** on CPU
2. **Created a small dataset** of Kenyan finance-related Q&A pairs  
3. **Generated predictions** from the model
4. **Computed ROUGE-L scores** using two approaches:
   * Hugging Face `evaluate` library for F1 scores
   * `rouge_score` library for detailed recall, precision, and F1 metrics

Here's the core evaluation code I used:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import evaluate
from rouge_score import rouge_scorer

# Load model and tokenizer
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

# Generate predictions
predictions = []
references = []

for example in dataset:
    prompt = example["question"]
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    predictions.append(generated_text)
    references.append(example["reference"])

# Compute ROUGE-L scores
rouge = evaluate.load("rouge")
results = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True,
    use_aggregator=False
)

# Detailed scoring
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = scorer.score(references[0], predictions[0])
```

## Example Results

Here's what I got from my first evaluation run:

```
ROUGE-L (F1): [0.375, 0.2]
ROUGE-L Recall: 0.2500
ROUGE-L Precision: 0.7500
ROUGE-L F1: 0.3750
```

## What I Learned

The results taught me several important things about LLM evaluation:

* **ROUGE-L recall** tells me how much of the reference content was covered by my model's output
* **Precision** shows how much of the generated text actually matches the reference  
* **F1** balances both metrics, giving me a single score to compare
* Even with a small dataset, these metrics help me see where the model is strong or weak

The first question got a decent F1 score of 0.375, while the second performed worse at 0.2. This suggests my model handles some finance topics better than others, which makes sense given the training data distribution.

## Technical Setup Notes

A few things I discovered during implementation:

* **CPU generation** worked fine for small batches, though GPU would be faster for larger evaluations
* **LoRA adapters** were properly loaded and functional in the evaluation pipeline
* **Tokenizer settings** like `pad_token_id` needed explicit configuration for generation
* Both evaluation libraries (`evaluate` and `rouge_score`) gave consistent results, which was reassuring

## Next Steps

This was a small but meaningful step in my LLM evaluation journey. Moving forward, I'm planning to:

1. **Expand the dataset** beyond just two examples for more reliable metrics
2. **Test other metrics** like BLEU, METEOR, and BERTScore to get a more complete picture
3. **Compare different model versions** to see how fine-tuning affects performance
4. **Explore semantic similarity metrics** that go beyond n-gram overlap

## Code and Resources

The complete implementation is available in my [GitHub repository](https://github.com/Okoth67/wekeza-llm-eval-rougel-1/tree/main). The notebook includes all the setup code, model loading, and evaluation logic.

For anyone starting their own LLM evaluation journey, I'd recommend beginning with ROUGE-L as it's straightforward to implement and interpret. The [original ROUGE paper](https://aclanthology.org/W04-1013.pdf) is also worth reading for the theoretical background.

## Acknowledgments

Thanks to Claude (Anthropic) for helping debug evaluation pipeline issues and providing insights on metric interpretation. Sometimes a second pair of eyes really helps when you're learning something new!

---

*Have you experimented with LLM evaluation metrics? What approaches have worked best for your use cases? Feel free to share your experiences!*
