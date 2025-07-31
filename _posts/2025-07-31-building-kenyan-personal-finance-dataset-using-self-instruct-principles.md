---
layout: post
title: "Building a Kenyan Personal Finance Dataset Using Self-Instruct Principles"
date: 2025-07-31
categories: [machine-learning, nlp, kenya, finance]
tags: [self-instruct, dataset, llm, fine-tuning, kenyan-finance]
---

# 📊 Building a Kenyan Personal Finance Dataset Using Self-Instruct Principles

🧠 **Inspired by the paper:** Self-Instruct: Aligning Language Models with Self-Generated Instructions  
💻 **GitHub Repo:** [Okoth67/1_upload_kenyan_finance_dataset](https://github.com/Okoth67/1_upload_kenyan_finance_dataset)

## 🔍 Background

The paper "Self-Instruct" by Wang et al. (2023) introduced an effective way to align large language models (LLMs) with human intent without human-labeled instruction data. Instead, it uses a bootstrapped pipeline:

1. **Generate diverse instructions** using a language model
2. **Use the same or another model** to generate responses to those instructions
3. **Filter for quality**, and fine-tune the model on this synthetic dataset

This method significantly improved performance on downstream tasks — even with just synthetic data.

## 🇰🇪 Applying the Method in a Kenyan Context

Inspired by this approach, I created a dataset tailored to Kenyan personal finance, aligning closely with the Self-Instruct method — but focused on local economic realities such as SACCOs, money market funds, government securities, inflation, budgeting in Nairobi, and more.

### ✅ Project Goals:
- Build a foundation for instruction-tuning LLMs in a Kenyan financial context
- Use models like GPT-3.5/4 to self-generate high-quality instruction–response pairs
- Format the data in `.jsonl` for future fine-tuning

## 🛠 My Implementation

📂 **Repo:** [https://github.com/Okoth67/1_upload_kenyan_finance_dataset](https://github.com/Okoth67/1_upload_kenyan_finance_dataset)  
📝 **Notebook:** `01_upload_kenyan_finance_dataset.ipynb`

I used a prompt-driven generation strategy across multiple models (e.g., GPT-4, Claude, etc.) to simulate the "self-generated" data aspect of the paper. The dataset was saved locally as:

```
C:/Users/bbollo/Downloads/kenyan_finance_dataset_v1.jsonl
```

### 🧾 Code Snippet:

```python
import json

dataset_path = "C:/Users/bbollo/Downloads/kenyan_finance_dataset_v1.jsonl"
data = []

with open(dataset_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping invalid JSON on line {i}: {e}")

print(f"📊 Total valid entries loaded: {len(data)}")

for i, entry in enumerate(data[:3], 1):
    print(f"\n--- Entry {i} ---")
    print("Instruction:", entry["instruction"])
    print("Response:", entry["response"])
```

### 🖨️ Sample Output:

```
⚠️ Skipping invalid JSON on line 1: Expecting ',' delimiter...
⚠️ Skipping invalid JSON on line 49: Expecting value...
📊 Total valid entries loaded: 47

--- Entry 1 ---
Instruction: How do I buy Treasury Bills and what's the current rate?
Response: You can buy Treasury Bills through your bank...

--- Entry 2 ---
Instruction: What should I look for when choosing a SACCO...
Response: When choosing a SACCO, consider...

--- Entry 3 ---
Instruction: Is it worth investing in the Nairobi Securities Exchange as a beginner?
Response: The NSE can be profitable long-term...
```

## 🔗 Connection to the Paper

This implementation reflects **Core Principles of Self-Instruct**:

| Paper Component | My Project Equivalent |
|---|---|
| Instruction generation using GPT-3 | Used GPT-based prompts to create Kenya-specific queries |
| Response generation from the same LM | Self-generated responses using the same model |
| No human labeling | All data generated synthetically from prompts |
| Alignment with real-world tasks | Focused on local financial literacy & investment advice |

While the original paper focuses on improving general LLM alignment, this project applies the concept to **domain-specific alignment** for Kenya's financial sector.

## 📌 What's Next?

- Convert the `.jsonl` dataset to Alpaca format for instruction-tuning with Mistral or DistilGPT2
- Train a local financial assistant model for Kenyan users (**Wekeza LLM**)
- Use this dataset to bootstrap future QLoRA or PEFT fine-tuning workflows
- Expand the dataset to 100+ entries and share it on Hugging Face Datasets

## 📁 Resources

🧾 **Paper:** [Self-Instruct (Wang et al., 2023)](https://arxiv.org/abs/2212.10560)  
💻 **Code:** [GitHub Repo](https://github.com/Okoth67/1_upload_kenyan_finance_dataset)  
🧠 **Related:** DistilGPT2 Wekeza LLM Finetuning Repo

## 💬 Final Thoughts

This mini-project proves how the Self-Instruct approach can be localized and leveraged for low-resource domains like Kenyan finance. It's a powerful way to bootstrap instruction data even without expensive human annotations — a great fit for specialized LLM projects like **Wekeza LLM**.

Let me know if you'd like to contribute or replicate this approach for your country, language, or industry!

---

## 🛠️ Tools Used

This blog post was enhanced using:
- **ChatGPT** for content summarization and structure optimization
- **Grammarly** for grammar checking and writing refinement

---

*Want to collaborate or have questions about this project? Feel free to reach out or contribute to the [GitHub repository](https://github.com/Okoth67/1_upload_kenyan_finance_dataset)!*
