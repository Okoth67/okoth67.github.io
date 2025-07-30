# Few-Shot Learning in Action: Testing GPT-3's Core Ideas with My Kenyan Financial Assistant

When OpenAI released their groundbreaking paper ["Language Models are Few-Shot Learners"](https://arxiv.org/pdf/2005.14165) in 2020, they demonstrated something remarkable: GPT-3 could learn new tasks from just a few examples in the prompt, without any parameter updates. This "in-context learning" ability suggested that large language models had developed an internal capacity to recognize patterns and generalize from minimal demonstrations.

But what about smaller, domain-specific models? Could a lightweight financial assistant exhibit similar few-shot capabilities?

## Enter Wekeza LLM

To explore this question, I experimented with **Wekeza LLM** , my custom financial assistant built on DistilGPT2 and fine-tuned using LoRA on Kenyan investment data. While GPT-3 boasts 175 billion parameters, Wekeza is intentionally compact, designed for specialized tasks like money market fund reporting and investment analysis.

The experiment was simple but revealing: test zero-shot, one-shot, and few-shot prompting strategies on the same financial summarization task.

## The Experiment

I set up three different prompting approaches:

**Zero-shot**: Just the instruction, no examples
```python
prompt_0shot = "Instruction: Summarize this week's Money Market Fund performance in Kenya.\nOutput:"
```

**One-shot**: A single example followed by the target task
```python
prompt_1shot = """Example:
Instruction: Summarize this week's performance of Cytonn Money Market Fund.
Output: Cytonn MMF delivered 11.2% annualized returns with stable liquidity and low risk.

Instruction: Summarize this week's Money Market Fund performance in Kenya.
Output:"""
```

**Few-shot**: Multiple examples before the final instruction
```python
prompt_fewshot = """Example 1:
Instruction: Provide a report on Sanlam Money Market Fund.
Output: Sanlam MMF posted 10.9% returns this week, maintaining steady growth.

Example 2:
Instruction: Report on the performance of NCBA MMF.
Output: NCBA MMF offered 11.0% returns, highlighting investor confidence.

Instruction: Summarize Kenya's MMF performance this week.
Output:"""
```

## What I Discovered

The results were fascinating and somewhat unexpected:

### Zero-Shot Performance
The model struggled without context, producing repetitive and irrelevant output focused on M-Pesa daily rates rather than money market funds.

### One-Shot Learning
With just one example, the model immediately improved! It correctly identified the task format and produced a coherent response about Cytonn MMF, though it became repetitive in extended generation.

### Few-Shot Challenges
Surprisingly, the few-shot prompt performed worse than one-shot, producing minimal and then repetitive output. This suggests that my smaller model may have reached its context processing limits with multiple examples.

## Key Insights

1. **Context Matters**: Even small models benefit significantly from in-context examples
2. **Sweet Spot**: For domain-specific smaller models, one good example might be more effective than multiple examples
3. **Model Limitations**: Unlike GPT-3, smaller models may struggle with complex few-shot scenarios due to limited context processing capabilities
4. **Domain Specialization**: Fine-tuning on specific data (Kenyan financial content) helped the model understand the task domain better

## The Broader Picture

This experiment reinforced that the principles from "Language Models are Few-Shot Learners" apply beyond massive models. Even compact, specialized LLMs can exhibit in-context learning abilities when properly guided. The key is understanding your model's limitations and designing prompts that work within those constraints.

For practitioners working with smaller models, this suggests focusing on high-quality single examples rather than overwhelming the model with multiple demonstrations.

## Resources

- 📊 **Full Experiment**: [GitHub Repository](https://github.com/Okoth67/wekeza-fewshot-prompting-gpt3style/tree/main)
- 📄 **Original Paper**: [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)

---

*This exploration was part of my ongoing journey in fine-tuning and deploying language models for real-world applications. Special thanks to Claude for debugging assistance and ChatGPT for helping structure this blog post and summary.*
