# Refining LLM Alignment with InstructZero: A Kenyan Finance Dataset

Building reliable instruction-following LLMs requires quality over quantity. This drove the creation of my latest dataset: `kenyan_finance_selfinstruct_v6_instructzero.jsonl`, applying [InstructZero](https://arxiv.org/abs/2306.03082) principles to filter unstable instructions.

## The Problem

Not all instructions are equal. Some produce inconsistent responses when paraphrased, making them risky for fine-tuning models in high-stakes domains like finance.

## The Solution

Following the [InstructZero methodology](https://arxiv.org/abs/2306.03082), I filtered my v5 dataset of 1,200+ instructions:

1. **Rewrote each instruction** into 2-3 paraphrases
2. **Generated responses** using `distilgpt2-wekeza-finetuned_v5_lora` 
3. **Compared outputs** and dropped instructions with semantic divergence

The result: a clean, consistency-validated dataset of ~800 instructions covering Kenyan investment topics like money market funds, SACCOs vs fixed deposits, CBK regulations, and portfolio diversification.

## Dataset Details

**[Wekeza v6 InstructZero Dataset](https://github.com/Okoth67/wekeza-dataset-instructzero-v6/tree/main)**
- 800+ filtered instruction-response pairs
- Domain-tuned filtering with LoRA model
- Focus on Kenyan financial reasoning and investment education

## What's Next

- Fine-tune `distilgpt2-wekeza-finetuned_v6_lora`
- Evaluate alignment robustness with IFScore
- Release chatbot demo and model card

This work builds on **InstructZero: Efficient Instruction Optimization for Black-Box Large Language Models** by Qiu et al., demonstrating how consistency filtering can improve synthetic datasets for specialized domains.

---

*Complete dataset available in the [GitHub repository](https://github.com/Okoth67/wekeza-dataset-instructzero-v6/tree/main).*
