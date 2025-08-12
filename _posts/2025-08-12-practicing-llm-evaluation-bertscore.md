# Practicing LLM Evaluation with BERTScore

*August 12, 2025*

As part of my early steps into LLM evaluation, I decided to explore BERTScore, a metric introduced in the paper "BERTScore: Evaluating Text Generation with BERT" by Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi ([link](https://arxiv.org/abs/1904.09675)).

## Understanding BERTScore's Approach

The motivation behind BERTScore is that traditional metrics like BLEU and ROUGE focus heavily on exact token matches, which can undervalue semantically correct but lexically different outputs. BERTScore instead leverages contextual embeddings from pre-trained transformer models (like BERT) to compare generated text and references on a semantic level, using cosine similarity to align tokens and calculate precision, recall, and F1 scores.

The authors demonstrate in their paper that BERTScore correlates much better with human judgment than traditional n-gram based metrics, particularly for tasks where paraphrasing and semantic equivalence matter more than exact word matching.

## My Implementation

In my notebook, I loaded a locally fine-tuned GPT-2 model, created a small custom dataset of questions and reference answers, generated model outputs and compared them to references using BERTScore, and observed how BERTScore captures semantic similarity even when wording differs significantly from the reference.

Here's my setup code:

```python
!pip install evaluate datasets transformers --quiet

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./distilgpt2-wekeza-finetuned_v5_cot_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.eval()
```

I created a simple test dataset with finance-related questions:

```python
from datasets import Dataset

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
```

The BERTScore evaluation required installing the specific library and using the evaluate framework:

```python
pip install bert_score

import evaluate
bertscore = evaluate.load("bertscore")
```

Here's the generation and evaluation process:

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
    references.append(example["reference"])

results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"
)

print(f"BERTScore Precision: {sum(results['precision'])/len(results['precision']):.4f}")
print(f"BERTScore Recall: {sum(results['recall'])/len(results['recall']):.4f}")
print(f"BERTScore F1: {sum(results['f1'])/len(results['f1']):.4f}")
```

## What I Learned from This Exercise

This hands-on exercise reinforced how embedding-based metrics can better reflect meaning over surface form, making them useful for evaluating modern generative models. Unlike the previous metrics I tested (ROUGE-L, BLEU, METEOR), BERTScore doesn't rely on exact word matches or even stemming and synonyms. Instead, it uses the rich contextual representations that BERT learned during pre-training.

The practical implementation taught me several important things about BERTScore. First, it requires specifying the language parameter, which makes sense given that different languages would need different BERT models for optimal performance. Second, it returns individual scores for each prediction-reference pair, allowing for more granular analysis than aggregate metrics alone.

While this is just practice, it's a valuable step in building intuition about the strengths and weaknesses of different evaluation approaches. BERTScore represents a significant evolution from traditional lexical metrics, incorporating the semantic understanding that modern transformer models have learned from large text corpora.

## Comparing with Previous Metrics

Having now tested ROUGE-L, BLEU, METEOR, and BERTScore on the same small dataset, I'm starting to see the progression in evaluation methodology. Each metric builds on the limitations of its predecessors. BLEU introduced automatic evaluation for translation, ROUGE adapted it for summarization, METEOR added stemming and synonyms, and now BERTScore brings semantic understanding through contextual embeddings.

The scores from BERTScore will likely be higher than my previous experiments with the same data, not because my model suddenly got better, but because BERTScore can recognize semantic similarity even when the exact wording differs. This aligns with what Zhang et al. demonstrated in their paper about correlation with human judgment.

## Technical Notes

The implementation was relatively straightforward thanks to the Hugging Face ecosystem. The `evaluate` library handles the complexity of loading the appropriate BERT model for computing embeddings, while the `bert_score` package provides the underlying functionality. The averaging of precision, recall, and F1 scores across predictions gives a summary view of performance.

One thing I noticed is that BERTScore is computationally heavier than the previous metrics I tested, which makes sense given that it needs to run inference through BERT to get contextual embeddings for both predictions and references.

## Moving Forward

These evaluation experiments have given me a solid foundation in understanding different approaches to measuring text generation quality. Each metric has taught me something different about the evolution of evaluation methodology in NLP. Next, I'm considering exploring some of the newer metrics that have emerged, or perhaps running a more comprehensive comparison across a larger dataset.

## Resources

The complete code for this experiment is available in my [GitHub repository](https://github.com/Okoth67/llm_eval_bertscore_intro/tree/main). The original [BERTScore paper](https://arxiv.org/abs/1904.09675) provides excellent background on the motivation and methodology behind using contextual embeddings for evaluation.

For anyone working through similar evaluation exercises, I'd recommend trying BERTScore after getting familiar with traditional metrics like BLEU and ROUGE. The contrast really helps illustrate how evaluation methodology has evolved alongside advances in language modeling.

## Acknowledgments

Thanks to Claude (Anthropic) for help with blog structure and organizing these evaluation experiments into coherent learning progression.
