# Building a Recursive Depth Dataset Inspired by Mixture-of-Recursions

*August 8, 2025*

I recently came across an fascinating paper on **Mixture-of-Recursions (MoR)** by Bae et al. from KAIST AI, Google, and Mila, which introduces a unified framework for efficient language models that combines parameter sharing with adaptive computation. The core idea is brilliant: use recursive transformers with dynamic depth allocation per token, allowing models to "think deeper" on complex tokens while processing simple ones efficiently.

## The MoR Innovation

The [original paper](https://arxiv.org/pdf/2507.10524) presents MoR as a solution to the computational challenges of scaling language models. Instead of using fixed computation for all tokens, MoR:

- **Reuses shared layer stacks** across recursion steps for parameter efficiency
- **Employs lightweight routers** to assign different recursion depths to individual tokens
- **Focuses attention computation** only on active tokens at each depth
- **Selectively caches** key-value pairs for memory efficiency

The results are compelling: MoR achieves better perplexity and few-shot accuracy compared to vanilla transformers while maintaining higher throughput.

## Inspired Implementation: Wekeza Depth Dataset

Reading this paper sparked an idea for my own project. While I can't implement the full MoR architecture, I could create a **depth-aware training dataset** that mimics the concept of adaptive thinking depth. 

The idea is simple: for each question, generate three response variants:
- **Depth 1**: Quick, direct answer
- **Depth 2**: Step-by-step reasoning  
- **Depth 3**: Detailed analysis with context

## The Implementation

Here's my Python script that creates this multi-depth dataset:

```python
import json
from pathlib import Path

output_path = Path("wekeza_depth_dataset.jsonl")

# Alpaca formatting
def make_alpaca_entry(instruction, output):
    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }

# Recursive Depth Function
def create_depth_versions(question, short_answer, reasoning_steps, detailed_analysis):
    """
    Given a question, create 3 depth versions for training.
    """
    return [
        make_alpaca_entry(f"<depth=1> {question}", short_answer),
        make_alpaca_entry(f"<depth=2> {question}", reasoning_steps),
        make_alpaca_entry(f"<depth=3> {question}", detailed_analysis)
    ]

# Example Seeds
seeds = [
    {
        "question": "How do I invest in a Kenyan Money Market Fund?",
        "short": "Choose a licensed fund, open an account, deposit money.",
        "reasoning": "1. Research licensed funds from CMA list. 2. Compare returns and fees. 3. Fill application form. 4. Fund your account.",
        "detailed": "Start by visiting the Capital Markets Authority (CMA) website to find a licensed provider. Compare annualized yields, minimum investment amounts, and management fees. Fill the application form (online or physical), attach your KYC documents (ID, proof of address), and deposit funds via bank transfer or mobile money. Many providers, such as CIC, Sanlam, and Britam, allow deposits from as little as KES 1,000. Reference: CMA Kenya, 2024."
    },
    {
        "question": "What are the risks of investing in Kenyan Treasury Bills?",
        "short": "Low credit risk, but interest rate and liquidity risks exist.",
        "reasoning": "1. T-Bills are backed by the government. 2. Low default risk. 3. Prices can be affected by interest rate changes. 4. Funds are locked until maturity.",
        "detailed": "Kenyan Treasury Bills are short-term government debt instruments, generally considered low-risk due to government backing. The main risks are: (1) Interest rate risk — if rates rise after purchase, your returns are comparatively lower. (2) Liquidity risk — your funds are tied until maturity unless sold in the secondary market. Reference: Central Bank of Kenya, 2024."
    }
]

# Generate the dataset
dataset = []
for s in seeds:
    dataset.extend(
        create_depth_versions(
            s["question"], 
            s["short"], 
            s["reasoning"], 
            s["detailed"]
        )
    )

with open(output_path, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset saved to {output_path} with {len(dataset)} examples.")
```

## Results and Applications

The script successfully generated a 6-example dataset with varying response depths. This approach could be valuable for:

1. **Training models** to provide responses at appropriate detail levels
2. **Fine-tuning** existing models for context-aware verbosity
3. **Creating educational datasets** that teach progressive complexity
4. **Building chatbots** that adapt response depth to user expertise

## Future Directions

While this is a simple proof-of-concept, it opens interesting possibilities:

- **Automatic depth detection**: Train models to predict optimal response depth
- **Curriculum learning**: Use depth progression for more effective training
- **Multi-domain expansion**: Apply this pattern across various knowledge domains
- **Interactive systems**: Allow users to request deeper explanations dynamically

## Code and Resources

The complete implementation is available in my [GitHub repository](https://github.com/Okoth67/wekeza-mor-depth-dataset/tree/main). Feel free to fork, extend, and experiment with your own depth-aware datasets.

The original MoR paper provides much deeper technical insights and is definitely worth reading for anyone interested in efficient transformer architectures.

## Acknowledgments

Special thanks to Claude (Anthropic) for help with blog structure and debugging the dataset generation code. Sometimes you need a good pair of AI eyes to spot the little things!

---
