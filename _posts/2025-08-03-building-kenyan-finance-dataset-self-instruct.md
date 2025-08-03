# Using Self-Instruct to Build a Kenyan Finance Dataset with LLMs

Recently, I came across the [Self-Instruct paper](https://arxiv.org/abs/2212.10560), which explores how large language models can be aligned to follow instructions without needing large amounts of human-annotated data. The idea is simple but powerful: prompt a language model to generate its own instruction-following examples, filter and clean the best ones, and then fine-tune a new model using that synthetic dataset.

## The Problem with Existing Data

I've been working on **Wekeza LLM**, a project focused on building a local AI assistant for answering Kenyan investment questions. Most publicly available datasets don't cover money market funds, Saccos, mobile investment apps, or financial reasoning in the Kenyan context. 

So, I decided to apply the Self-Instruct method to generate my own dataset for this use case.

## Building the Dataset

Using a small open-source language model, I generated over 60 high-quality instruction-input-output examples covering everything from budgeting advice to comparisons between T-bills and Saccos. Each entry was reviewed for clarity, diversity, and realism.

The examples span real scenarios that Kenyan investors face: choosing between Sacco dividends and bank deposits, understanding mobile money investment options, comparing treasury bills with money market funds, and getting practical budgeting advice tailored to local economic conditions.

The full dataset is now available here: [Self-Instruct Kenyan Finance v5](https://github.com/Okoth67/selfinstruct-kenyan-finance-v5).

## What's Next

This dataset will be used to fine-tune the next version of Wekeza LLM. The goal is to make it more helpful, more aligned with real user needs, and capable of answering nuanced financial questions in a local context even without access to large proprietary data.

The Self-Instruct approach has turned out to be a great fit for this kind of grassroots, domain-specific LLM development. It's low-cost, practical, and scales well with creativity. Instead of waiting for someone else to build datasets that capture Kenyan financial realities, I could create exactly what was needed.

## Lessons Learned

Building AI for underserved contexts doesn't have to mean waiting for big tech companies or expensive data collection projects. With techniques like Self-Instruct, you can bootstrap quality training data that reflects the specific needs and nuances of your target users.

The process taught me that sometimes the best way forward is to build the tools you wish existed. Now other developers working on African fintech or similar local AI projects have a concrete example and dataset to reference.

**Resources:**
- [Self-Instruct Paper](https://arxiv.org/abs/2212.10560)
- [Kenyan Finance Dataset](https://github.com/Okoth67/selfinstruct-kenyan-finance-v5)
