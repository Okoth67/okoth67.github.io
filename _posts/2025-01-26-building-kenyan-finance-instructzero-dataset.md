# Building a Zero-Shot Instruction Dataset for Kenyan Finance — Inspired by InstructZero

Instruction tuning has revolutionized how we align language models to follow user commands. Yet, much of this progress has focused on high-resource domains and languages. In the Kenyan financial space covering SACCOs, money market funds, and fixed deposits, high-quality, instruction-following data is almost non-existent.

To fill this gap, I've created a dataset titled **Kenyan Finance InstructZero v6**, inspired by the InstructZero methodology. This approach leverages only a base instruction-tuned LLM to generate high-quality instruction-response examples without any human-written prompts or answers. It's a fully zero-shot framework that produces aligned data in low-resource domains like ours.

## Why InstructZero Works for This Problem

The InstructZero methodology explores how you can bootstrap aligned instruction data from scratch, without human annotations or prompt templates. This makes it perfect for building training datasets in emerging domains like Kenyan financial literacy where curated data is scarce.

Traditional instruction tuning requires extensive human-authored examples, which are expensive and time-consuming to create, especially for domain-specific applications. The zero-shot approach eliminates this bottleneck by having the model generate its own training data based on minimal seed instructions.

## My Adaptation: Wekeza v6

Using this zero-shot approach, I sampled realistic instructions covering essential aspects of Kenyan personal finance:

**Investment Comparisons**: Analyzing money market funds versus fixed deposits in the Kenyan context  

**Inflation Impact**: Understanding investment tradeoffs under Kenya's economic conditions  

**SACCO Analysis**: Exploring risks and benefits of Savings and Credit Cooperative Organizations

**Practical Advice**: Saving strategies tailored for low-income earners in Kenya

**Tax Guidance**: Navigating tax implications of various Kenyan financial products

**Market Insights**: Understanding local investment vehicles and their performance

The model generated coherent, actionable responses that reflect real financial decisions faced by ordinary Kenyans. All examples follow the Alpaca instruction format, making them compatible with popular fine-tuning frameworks. This became the foundation for version 6 of my ongoing project: **Wekeza LLM**, a locally-focused instruction-following model for Kenyan investors.

## What's in the Dataset?

**Format**: JSONL (each line contains a complete instruction example)

**Generation Method**: Purely model-generated, filtered for quality and relevance

**Content Focus**: Real-world financial reasoning tasks faced by Kenyans

**Language**: Primarily English with occasional Swahili financial terms

**Quality Control**: Manual review and filtering to ensure accuracy and cultural relevance

The dataset covers scenarios from basic savings advice to complex investment analysis, all grounded in Kenya's unique financial landscape. No human-written instructions or outputs were used. The entire dataset emerged from the zero-shot generation process.

**Repository**: [Kenyan Finance InstructZero v6](https://github.com/Okoth67/kenyan-finance-instructzero-v6/tree/main)

## What's Next?

The immediate goal is to fine-tune a compact model like Mistral-7B or Falcon-1B using LoRA or QLoRA techniques on this dataset. The plan includes several phases:

First, I'll focus on model training using parameter-efficient methods. Then comes evaluation, where I'll benchmark performance on realistic Kenyan financial instruction-following tasks. Based on those results, I'll iterate by expanding the dataset to address any performance gaps. Finally, the goal is community impact by making the model accessible for financial education in Kenya.

The evaluation will specifically test the model's ability to provide accurate, culturally-appropriate financial guidance, especially in domains that mainstream instruction-tuned models typically ignore or handle poorly due to their Western-centric training data.

## Community and Collaboration

This work addresses a significant gap in AI alignment for emerging markets. If you're building aligned models for underrepresented domains, African markets, or code-mixed languages, I'd love to connect and collaborate.

The dataset is open-source and designed to help researchers and practitioners working on low-resource instruction tuning, financial AI for emerging markets, Swahili-English code-mixed applications, and domain-specific model alignment.

*Blog structure inspired by ChatGPT's organizational framework. Special thanks to the instruction tuning community for making this work possible.*

## References

InstructZero methodology and framework concepts

[Kenyan Finance InstructZero v6 Dataset](https://github.com/Okoth67/kenyan-finance-instructzero-v6/tree/main)

FLAN: Finetuned Language Models Are Zero-Shot Learners (Wei et al.)

*This project represents a step toward democratizing AI alignment for underserved communities and financial contexts. Every dataset contribution helps bridge the gap between advanced AI capabilities and real-world applications in emerging markets.*
