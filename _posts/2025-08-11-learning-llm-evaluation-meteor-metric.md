# Learning LLM Evaluation with METEOR

*August 11, 2025*

As part of my ongoing journey into **LLM evaluation**, I explored the **METEOR** metric, a method introduced by Banerjee and Lavie in their 2005 paper *"METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments"* ([link](https://aclanthology.org/W05-0909.pdf)).

## Understanding METEOR's Design

The METEOR paper was developed in the context of **machine translation**, aiming to address limitations of BLEU by incorporating not just exact word matches, but also **stem matches**, **synonyms**, and **word order**. This design makes METEOR more sensitive to semantic equivalence, often aligning better with human judgments.

Unlike BLEU's strict n-gram matching, METEOR uses WordNet to identify synonymous words and Porter stemming to match different word forms. The authors demonstrated that this approach correlates better with human evaluation, particularly for longer texts where paraphrasing is common.

## My Practical Implementation

In my notebook, I applied METEOR to a small set of financial Q&A pairs. Using my fine-tuned GPT-2 model (*distilgpt2-wekeza-finetuned_v5_cot_lora*), I generated answers to predefined questions, compared them with reference answers, and calculated the METEOR score using Hugging Face's `evaluate` library.

Here's the core implementation:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
import evaluate

# Load the fine-tuned model
model_path = "./distilgpt2-wekeza-finetuned_v5_cot_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.eval()

# Sample dataset
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
    references.append([example["reference"]])

# Compute METEOR score
meteor = evaluate.load("meteor")
results = meteor.compute(predictions=predictions, references=references)
print(f"METEOR Score: {results['meteor']:.4f}")
```

## Results and Observations

The experiment yielded a **METEOR score of 0.1630**, which was notably higher than the BLEU score (0.0223) I got from the same dataset in my previous experiment. This difference illustrates exactly what Banerjee and Lavie discussed in their paper: METEOR's ability to capture semantic similarity beyond exact word matches.

During setup, I noticed METEOR automatically downloaded several NLTK resources: **WordNet** for synonym matching, **Punkt tokenizer** for sentence segmentation, and **Open Multilingual Wordnet** for extended language support.

This automatic resource management shows how the `evaluate` library handles METEOR's dependencies, making the original paper's complex linguistic processing accessible through a simple API.

## Technical Insights

While the dataset was tiny and purely illustrative, the process gave me hands-on experience in running inference on a custom LLM with LoRA adapters, preparing predictions and references for evaluation in the expected format, understanding how METEOR interprets semantic similarity compared to purely lexical metrics, and observing automatic dependency management for linguistic resources.

The higher METEOR score compared to BLEU suggests that my model's responses, while not exact matches, contained semantically related content that METEOR's stemming and synonym matching could recognize.

## Connecting Theory to Practice

This was not a full-fledged research experiment, but rather **a practical first step** in connecting academic evaluation metrics to real-world model testing. The original METEOR paper's focus on correlation with human judgment becomes much more tangible when you see how it treats paraphrases and synonyms differently than BLEU.

It's clear that even simple experiments like this help bridge the gap between reading a paper and applying its ideas to an actual model. METEOR's design philosophy of incorporating linguistic knowledge rather than relying purely on surface-level matches makes it particularly relevant for evaluating conversational AI systems.

## Next Steps in My Evaluation Journey

Moving forward, I'm planning to compare METEOR with BERTScore to see how embedding-based metrics perform, expand the test dataset for more reliable comparative analysis, test different generation parameters to see how they affect METEOR scores, and explore METEOR's individual components like precision, recall, and fragmentation penalty.

## Resources

The complete code for this experiment is available in my [GitHub repository](https://github.com/Okoth67/llm_eval_meteor_practice/tree/main). The original [METEOR paper](https://aclanthology.org/W05-0909.pdf) by Banerjee and Lavie provides excellent background on the metric's design principles and evaluation methodology.

For anyone interested in LLM evaluation, METEOR represents an important step between simple n-gram overlap metrics and modern embedding-based approaches. Understanding its linguistic foundations helps appreciate why correlation with human judgment matters in evaluation design.

## Acknowledgments

Thanks to Claude (Anthropic) for assistance with blog structure and debugging the evaluation pipeline. Having an AI collaborator for organizing thoughts and catching implementation details has been genuinely helpful in this learning process.

---

*Have you experimented with METEOR for evaluating language models? How do you find it compares to other metrics in your use cases? I'd love to hear about your evaluation experiences!*
