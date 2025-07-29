---
layout: post
title: "LoRA Fine-Tuning DistilGPT2 with Backtranslated Instructions for Wekeza LLM (v3)"
date: 2025-07-29 06:00:00 +0300
categories: finetuning wekeza LLM nlp
---

I'm a self-taught NLP and LLM student, learning by building hands-on projects. This blog is my personal learning diary. In this post, I document my LoRA fine-tuning experiment on `distilgpt2` using a small, backtranslated dataset generated with the Self-Alignment with Instruction Backtranslation method. This is the third version (`v3`) of my **Wekeza LLM**, a Kenya-specific investment assistant.

## 🔗 Paper Reference

The dataset was created using this recent paper:  
**[Self-Alignment with Instruction Backtranslation (2024)](https://arxiv.org/pdf/2507.16003)**  

It proposes a simple and effective method to synthetically generate instructions from outputs using the base model, improving alignment without requiring human labels.

---

## 📘 Project Summary

| Version | Dataset Source                | Examples | Method                     |
|---------|-------------------------------|----------|----------------------------|
| v1      | Human-written instructions    | 5        | Full fine-tune             |
| v2      | Self-Instruct (DistilGPT2)    | 5        | Fine-tune                  |
| v3      | Backtranslated (from v2)      | 5        | LoRA fine-tune (this post) |

---

## 🔧 Project Setup

This repo contains the training notebook:  
📁 [GitHub Notebook](https://github.com/Okoth67/distilgpt2-lora-finetune-wekeza-v3/tree/main)

### Dependencies

```bash
pip install peft transformers datasets accelerate bitsandbytes
```

### Model Base

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
```

## 📁 Dataset: WekezaLLM_backtranslated_v3.jsonl

Here's a sample entry in Alpaca-style format:

```json
{
  "instruction": "Jinsi ya kuwekeza katika soko la fedha la Kenya?",
  "input": "",
  "output": "Ili kuwekeza katika soko la fedha la Kenya, tafuta kampuni ya usimamizi wa fedha, fungua akaunti, weka pesa zako, na chagua mfuko wa fedha unaofaa."
}
```

You can load the dataset using HuggingFace Dataset class:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="WekezaLLM_backtranslated_v3.jsonl", split="train[:4]")
```

## 🏋🏾‍♂️ LoRA Fine-Tuning Code

### LoRA Configuration

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

### Training Arguments & Trainer

```python
training_args = TrainingArguments(
    output_dir="./distilgpt2-wekeza-finetuned_v3_lora_backtranslated",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    learning_rate=2e-4,
    logging_dir="./logs",
    save_steps=10,
    logging_steps=5,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
```

## ✅ Generation Sample

Here's how the model performed after fine-tuning:

**Prompt (instruction only):**
```
"Eleza jinsi ya kuanza kustaafu mapema nchini Kenya."
```

**Generated Output:**
```
"Ili kuanza kustaafu mapema, hakikisha unaanza kuwekeza mapema kupitia mifuko ya fedha, pensheni binafsi, na mipango ya akiba ya muda mrefu."
```

## 🧠 Reflections

This was a small but powerful experiment:

- **LoRA** let me fine-tune a base model quickly and with minimal resources.
- The **backtranslated dataset** closely mimicked real Kenyan financial queries.
- I've now built and compared **three different versions** of WekezaLLM.

**Next steps:** I'll scale the dataset using automated generation + quality filters.

## 🔗 Repository

All code and notebook files:  
📂 [GitHub Repo — distilgpt2-lora-finetune-wekeza-v3](https://github.com/Okoth67/distilgpt2-lora-finetune-wekeza-v3)

---

Stay tuned — tomorrow I'll post about expanding this dataset to 100+ examples using a chain of backtranslation and filtering.

Thanks for reading!

**Brian Bollo** — Building Wekeza LLM, one small experiment at a time.
