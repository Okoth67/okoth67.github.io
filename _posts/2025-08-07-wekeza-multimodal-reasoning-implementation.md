# Implementing Mixture of Thought for Kenyan Finance: A Multimodal Reasoning Experiment

*August 7, 2025*

## Introduction

While working on my Wekeza finance chatbot, I stumbled across an interesting paper by Zheng et al. on ["Learning to Reason via Mixture of Thought for Logical Reasoning"](https://arxiv.org/pdf/2505.15817). The central idea caught my attention: humans don't just think in natural language when solving problems. We sketch diagrams, write pseudocode, make tables, and switch between different mental models depending on what works best for the situation at hand.

Most language models, however, get trained on just one reasoning style at a time, typically natural language explanations. This seems like a missed opportunity, especially for financial reasoning where you might need plain explanations, mathematical calculations, and structured comparisons all working together.

## What MoT Actually Does

The paper's approach is straightforward but clever. Instead of forcing models to reason in just one way, they train on three different modalities:

**Natural language reasoning** works well for intuitive explanations and connecting concepts, but can get tangled up in complex logical chains.

**Code based reasoning** excels at computational steps and mathematical operations, though it can miss nuanced contextual factors.

**Truth table reasoning** systematically works through different scenarios, which helps catch edge cases that natural language might gloss over.

The researchers found that combining these approaches yielded significant improvements, with some benchmarks showing over 11 point gains in accuracy. More importantly, the different modalities seemed to complement each other's weaknesses.

## Adapting This for Financial Reasoning

After reading the paper, I decided to try adapting their approach for my Kenyan finance dataset. The process turned out to be more interesting than I expected.

I started with about 90 financial questions from my existing dataset, things like "How do I start investing in a money market fund in Kenya?" The original answers were solid but single dimensional. So I experimented with generating multiple reasoning paths for each question.

For each financial query, I prompted my model to respond in three different ways:

```python
def generate_modal_rationales(prompt, model, tokenizer):
    nl_prompt = f"{prompt}\nExplain in plain language:"
    code_prompt = f"{prompt}\nWrite a Python function to reason this out:"
    table_prompt = f"{prompt}\nMake a truth-table style comparison:"
```

The natural language responses handled the contextual aspects well, explaining regulatory requirements and cultural considerations for Kenyan investors. The code responses focused on calculations, interest computations, and logical decision trees. The table format proved surprisingly useful for comparing different investment options or outlining step by step processes.

Not every response was usable though. I had to filter out responses that were too short, code snippets without actual functions, and tables that didn't maintain proper structure. This filtering step was crucial because low quality examples would have polluted the training data.

## What I Learned from the Experiment

The initial results were encouraging, though I'm still in early stages. From just 3 test examples, I managed to generate coherent multimodal rationales that actually complemented each other. The natural language explanations provided context about Kenyan financial regulations, while the code versions focused on mathematical aspects like compound interest calculations and risk assessments.

What struck me most was how the different modalities revealed different aspects of the same financial questions. A query about money market funds might get a regulatory focused natural language response, a return calculation focused code response, and a structured comparison table showing different fund options.

The fine tuning process used LoRA (Low Rank Adaptation) on DistilGPT2, which kept things computationally manageable while still allowing the model to learn from these multimodal examples. I tagged each training example with modality indicators so the model could learn when to deploy different reasoning styles.

One challenge I encountered was ensuring quality across all three modalities. Not every financial question lends itself equally well to code based reasoning, and some topics work better with tables than others. The filtering mechanisms helped, but there's definitely room for improvement in how I generate and select the best examples for each modality.

## Looking Ahead

This experiment represents just the beginning of exploring multimodal reasoning for financial applications. The full MoT framework includes sophisticated inference mechanisms that I haven't implemented yet, where the model can dynamically choose or combine different reasoning modalities based on the problem at hand.

The complete code for this experiment is available in my [GitHub repository](https://github.com/Okoth67/wekeza_v6_lora_multimodal_cot_finetuning). I'm planning to expand the training dataset beyond the initial 90 examples and implement more of the original paper's inference strategies.

Financial reasoning seems particularly well suited to multimodal approaches because real financial decisions often require multiple perspectives. You need intuitive understanding of market dynamics, mathematical analysis of returns and risks, and systematic comparison of different options. Training models to seamlessly switch between these modes of thinking could lead to more robust financial advisory systems.

## Credits

Thanks to ChatGPT for helping debug some tricky data processing issues during implementation, and to Claude for assistance with structuring this writeup. The original Mixture of Thought framework by Zheng et al. provided the theoretical foundation that made this experiment possible.

The journey of building better reasoning capabilities for financial AI continues. Each experiment teaches something new about how these models learn and how we can guide them toward more human like reasoning patterns.

---

*Part of my ongoing exploration into advanced reasoning techniques for financial applications. More experiments and insights coming as I continue developing the Wekeza platform.*
