---
layout: post
title: "Benchmarking Wekeza LLM with IFScore"
date: 2025-01-25
categories: [machine-learning, finance, kenya]
tags: [llm, benchmark, distilgpt2, ifscore]
---
# Benchmarking Wekeza LLM with IFScore: Measuring Kenyan Investment Report Generation

## 📌 Problem Statement

As part of developing Wekeza LLM—a fine-tuned version of DistilGPT2 designed to generate concise Kenyan investment reports—I wanted to evaluate how effectively the model integrates domain-specific financial concepts in its outputs.

Inspired by the paper [**IFScore: Measuring How Well LLMs Follow Instructions Using Keyword Grounding**](https://arxiv.org/abs/2507.11538), I recreated and adapted the IFScore benchmark to measure how well my model follows investment prompts containing key Kenyan financial terms like "mpesa", "nssf", and "chama".

## 📚 Background: The IFScore Research

The [IFScore paper](https://arxiv.org/abs/2507.11538) addresses a critical challenge in modern LLM deployment: **how many instructions can models actually handle before performance meaningfully degrades?** As production systems increasingly require LLMs to follow dozens or hundreds of simultaneous instructions—from content guidelines to compliance standards—understanding these limitations becomes essential for reliable operation.

The researchers introduced IFScale, a benchmark using 500 keyword-inclusion instructions for business report writing, revealing that even frontier models achieve only 68% accuracy at maximum instruction density. Their findings show distinct performance degradation patterns based on model size and reasoning capability, with models exhibiting bias towards earlier instructions.

For my Wekeza LLM project, this research is particularly relevant because Kenyan investment reports must integrate multiple domain-specific concepts simultaneously. While the original IFScore focuses on high-density instruction following, I adapted it to evaluate how well my smaller, fine-tuned model grounds specific financial keywords in generated content—essentially measuring instruction adherence at a more focused scale.

## 💡 What is IFScore?

The IFScore metric computes how well a model-grounded response contains the exact financial concepts (keywords) that were part of the prompt. This is especially useful for models meant to generate structured responses in niche domains like finance.

It calculates:

**IFScore = (# of prompt keywords present in response) / (# of prompt keywords)**

## 🔍 Why This Matters for Domain-Specific Models

The challenge identified in the IFScore research—that models struggle with multiple simultaneous instructions—is particularly acute for specialized domains like Kenyan finance. Investment reports must accurately incorporate various financial instruments (unit trusts, money markets), regulatory bodies (CMA, CBK), and local payment systems (M-Pesa, chamas) while maintaining coherent, professional language.

By adapting IFScore for my use case, I could measure whether my fine-tuned DistilGPT2 model successfully grounds these domain-specific terms, even when multiple concepts need to be addressed in a single generation. This evaluation helps validate the model's readiness for real-world investment advisory applications.

## 🔧 Step-by-Step Implementation

### ✅ Step 1: Set Up the Environment

We loaded our fine-tuned DistilGPT2 model using Hugging Face's transformers pipeline:

```python
from transformers import pipeline

model_name = "distilgpt2-wekeza-finetuned_v1"
generator = pipeline("text-generation", model=model_name)
```

### Step 2: Define Prompts and Keywords

We used realistic prompts and a growing list of Kenyan financial keywords:

```python
prompt_templates = [
    "Write a professional investment update that includes insights on {}.",
    "Generate a financial advisory note for a client based on {}.",
    "Briefly explain the importance of {} in Kenyan investment.",
]

keywords = ["mpesa", "chama", "nssf", "unit trust", "money market", "bonds", "cma", "cbk", "risk", "interest", "capital", "liquidity", "returns"]
```

### ✍️ Step 3: Generate Reports and Track Data

For each group of 1 to 5 keywords, we generated 5 prompts and 3 responses per prompt. This gave us 75 generations total.

```python
import random

generation_data = []
for num_keywords in range(1, 6):
    for _ in range(5):
        selected_keywords = random.sample(keywords, num_keywords)
        keyword = random.choice(selected_keywords)
        prompt = random.choice(prompt_templates).format(keyword)

        responses = []
        for _ in range(3):
            output = generator(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.95
            )
            responses.append(output[0]["generated_text"])

        generation_data.append({
            "prompt": prompt,
            "keywords": selected_keywords,
            "responses": responses
        })
```

✔️ **Successfully generated 25 prompts × 3 responses = 75 generations.**

### 📊 Step 4: Compute IFScore for Each Output

We computed the score for each response based on how many of the prompt's keywords appeared in the text.

```python
from collections import defaultdict
import re

def count_keyword_occurrences(text, keywords):
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    return sum(1 for kw in keywords if kw.lower() in clean_text.split())

scores_by_keyword_count = defaultdict(list)

for example in generation_data:
    responses = example["responses"]
    keyword_count = len(example["keywords"])
    
    for response in responses:
        count = count_keyword_occurrences(response, example["keywords"])
        ifscore = count / keyword_count
        scores_by_keyword_count[keyword_count].append(ifscore)

average_ifscores = {
    k: sum(v) / len(v) if len(v) > 0 else 0
    for k, v in scores_by_keyword_count.items()
}
```

### 📈 Step 5: Visualize the IFScore

We plotted IFScore vs number of keywords in prompt:

```python
import matplotlib.pyplot as plt

x = sorted(average_ifscores.keys())
y = [average_ifscores[k] for k in x]

plt.figure(figsize=(8, 5))
plt.plot(x, y, marker="o", linestyle="-")
plt.title("IFScore vs Keyword Count in Prompt")
plt.xlabel("Number of Keywords in Prompt")
plt.ylabel("Average IFScore")
plt.grid(True)
plt.show()
```

## 🧪 Results

```
1 keyword(s): IFScore = 0.867
2 keyword(s): IFScore = 0.500
3 keyword(s): IFScore = 0.333
4 keyword(s): IFScore = 0.183
5 keyword(s): IFScore = 0.173
```

✅ As expected, IFScore drops as the number of keywords in the prompt increases—highlighting the challenge of grounding multiple concepts in a single generation.

## 🎓 Learning Outcomes

This project helped me:

- **Understand how to benchmark instruction-following** in text generation models
- **Apply a real research metric (IFScore)** to a local use-case (Kenyan investment)
- **Evaluate model grounding and relevance** in domain-specific generation tasks
- **Learn Python techniques** for evaluation loops, scoring, and plotting
- **Appreciate the trade-offs** between creativity (sampling) and accuracy (grounding)

---

*This post demonstrates the practical application of academic benchmarking methods to evaluate domain-specific language models, particularly in the context of Kenyan financial services.*

## 📝 Writing Notes

*Like any good data scientist, I believe in using the right tools for the job. This blog post was enhanced with AI assistance—Grammarly helped polish the grammar, and ChatGPT helped structure the narrative flow. Because let's be honest, if you're building LLMs, you might as well use them to write about building LLMs! 🤖✍️*
