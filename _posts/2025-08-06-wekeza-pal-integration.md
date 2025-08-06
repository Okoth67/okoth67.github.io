# From Chain of Thought to Program Aided Reasoning: Enhancing Wekeza LLM with PAL

One of the most exciting evolutions in reasoning with language models is the shift from pure natural language outputs to code-based solutions. The paper "Program-Aided Language Models (PAL)" introduced a powerful idea: instead of relying entirely on a language model to both understand and solve complex problems, we can delegate the actual computation to an interpreter like Python while the LLM focuses on translating the natural language question into runnable code.

This approach not only enhances reliability and accuracy but also enables symbolic precision in domains where even small errors matter, particularly in finance.

## Why This Matters for Wekeza

Wekeza LLM is designed as a Kenyan investment assistant, helping users navigate questions around savings, interest rates, and investment returns. These are often math heavy tasks, where small calculation errors can undermine trust in the assistant.

Inspired by PAL, I've integrated this reasoning approach into Wekeza's workflow. The model interprets a user's question and generates a Python program to solve it, then that program is executed directly in the notebook. This enables Wekeza to provide accurate answers to finance related questions involving interest rates, time horizons, savings targets, and more.

## Human Like Understanding, Machine Level Precision

Rather than relying on the LLM to do all the math internally (which often leads to arithmetic mistakes), PAL allows us to combine the interpretive strength of the LLM with the calculative accuracy of Python.

This hybrid model enables Wekeza to interpret natural language investment goals, break them into logical steps (like Chain of Thought), convert those steps into executable code, and deliver precise, verifiable answers.

## Early Results

Even with a small, fine tuned model like distilgpt2, PAL has shown promising results on financial reasoning tasks in Wekeza. With carefully crafted prompts, the model can now handle real world queries like "How much should I save monthly to reach 500,000 KES in 18 months with 10% annual interest?" and return answers backed by actual calculations, not guesswork.

## Looking Ahead

By fusing PAL with domain specific instruction tuning and chain of thought reasoning, Wekeza is evolving from a simple text generator into a reasoning financial agent. The future lies in pushing this even further by adding verification steps, fallback checks, and dynamic prompt templates that adapt to user queries.

The PAL paper didn't just introduce a new way to use LLMs. It offered a blueprint for domain specialized agents like Wekeza to become more trustworthy, more intelligent, and more useful.

## Implementation and Resources

Want to explore the implementation? Check out the Wekeza PAL Reasoning repository at https://github.com/Okoth67/wekeza_pal_reasoning_v1/tree/main to see how we've integrated program aided reasoning into a domain specific financial assistant.

The original research paper "Program-Aided Language Models" is available at https://arxiv.org/pdf/2211.10435

Special thanks to Claude for helping refine this post and providing insights during the development process.

What's your experience with program aided reasoning in specialized domains? I'd love to hear about your experiments and thoughts on this approach.
