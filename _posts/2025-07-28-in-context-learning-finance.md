# Using In-Context Learning to Predict Kenyan Money Market Returns

## 🧠 Experimenting with In-Context Learning on Money Market Funds

As part of my self-taught journey into LLMs and instruction learning, I wanted to test how well a small model (`distilgpt2`) can **predict investment returns** using **In-Context Learning (ICL)** with no further fine-tuning required.

This blog post explores a simple experiment based on the paper [Large Language Models Can Self-Instruct in Context (2025)](https://arxiv.org/pdf/2507.16003). The paper investigates how language models can **generate and learn from examples within the prompt**, i.e., in context without parameter updates. I applied this idea to the Kenyan finance domain using money market fund returns as my task.

📂 Full repo & notebook here: [GitHub - ICL-Kenyan-MoneyMarket-Returns](https://github.com/Okoth67/ICL-Kenyan-MoneyMarket-Returns/tree/main)

---

## 🛠️ The Setup

I used a fine-tuned `distilgpt2` model trained on Kenyan investment phrases (from earlier work). The idea is simple: show the model a few examples of fund returns, then ask it to complete a similar one.

### Loading the Model

First, I loaded my fine-tuned DistilGPT-2 model with LoRA adaptations:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./distilgpt2-wekeza-finetuned_v3_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

The model architecture shows LoRA (Low-Rank Adaptation) layers integrated into the attention and MLP components, which allows for efficient fine-tuning while maintaining the base model's capabilities.

### Text Generation Function

I created a simple generation function to handle the inference:

```python
def generate_output(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 🔎 Zero-Shot Performance (No In-Context Examples)

First, I tested the model's baseline performance without any examples:

```python
no_icl_prompt = "What is the return of CIC Money Market Fund?"
no_icl_output = generate_output(no_icl_prompt)
print("=== Non-ICL Prediction ===")
print(no_icl_output)
```

**Result:**
```
What is the return of CIC Money Market Fund?

### Response:
CIC Money Market Fund returns annually from CIC Money Market Fund to CIC Money Market Fund. 
CIC Money Market Fund returns annually from CIC Money Market Fund to CIC Money Market Fund...
```

As expected, the model without context produces repetitive, incoherent text that doesn't provide meaningful information about actual returns.

---

## 🎯 In-Context Learning: Structured Examples

Next, I provided the model with structured examples of Kenyan money market fund returns:

```python
icl_prompt = """
Below are examples of returns for Kenyan Money Market Funds:

Example 1:
Fund: NCBA Money Market Fund
Return: 13.5%

Example 2:
Fund: Sanlam Money Market Fund
Return: 13.3%

Example 3:
Fund: CIC Money Market Fund
Return:"""

icl_output = generate_output(icl_prompt)
print("=== ICL Prediction ===")
print(icl_output)
```

**Result:**
```
Below are examples of returns for Kenyan Money Market Funds:

Example 1:
Fund: NCBA Money Market Fund
Return: 13.5%

Example 2:
Fund: Sanlam Money Market Fund
Return: 13.3%

Example 3:
Fund: CIC Money Market Fund
Return: 12.5%

Example 4:
Money Market Fund
Return: 12.5%
...
```

The model successfully completed the pattern and predicted a 12.5% return for CIC Money Market Fund, which is reasonable given the context of the other funds' performance (13.5% and 13.3%). However, it continued generating additional examples, showing it learned the pattern but struggled with stopping at the appropriate point.

---

## 🗣️ Natural Language Prompting

I also experimented with a more natural language approach:

```python
icl_prompt_natural = """
Here are some summaries of returns from popular Kenyan money market funds:

- The NCBA Money Market Fund posted an annual return of 13.5% this quarter.
- Sanlam's Money Market Fund achieved a yield of 13.3% during the same period.
- CIC Money Market Fund recorded a return of
"""

icl_output_natural = generate_output(icl_prompt_natural)
print("=== ICL Prediction (Natural Language Style) ===")
print(icl_output_natural)
```

**Result:**
```
Here are some summaries of returns from popular Kenyan money market funds:

- The NCBA Money Market Fund posted an annual return of 13.5% this quarter.
- Sanlam's Money Market Fund achieved a yield of 13.3% during the same period.
- CIC Money Market Fund recorded a return of 12.5% in 2015.
```

Interestingly, the model provided a similar 12.5% prediction but added "in 2015" as a time reference, showing it can adapt to different prompt styles while maintaining consistency in its predictions.

---

## 📊 Key Observations

### What Worked Well:
1. **Pattern Recognition**: The model successfully identified the pattern in the examples and generated a reasonable return prediction (12.5%) that fell within the expected range.

2. **Context Adaptation**: The model showed ability to adapt to different prompt formats (structured vs. natural language) while maintaining consistent predictions.

3. **Domain Knowledge**: The fine-tuned model demonstrated understanding of Kenyan financial terminology and fund names.

### Limitations:
1. **Repetitive Generation**: The model continued generating examples beyond what was needed, indicating challenges with appropriate stopping.

2. **No Real-Time Data**: The predictions are based on learned patterns rather than actual current market data.

3. **Limited Reasoning**: The model follows patterns but doesn't demonstrate deeper financial reasoning about why certain returns might be expected.

---

## 🔬 Connection to the Research

This experiment aligns with findings from the paper ["Large Language Models Can Self-Instruct in Context"](https://arxiv.org/pdf/2507.16003), which explores how language models can generate and learn from examples within the prompt without parameter updates. The paper demonstrates that models can effectively use in-context examples to improve performance on various tasks.

In my experiment, the DistilGPT-2 model showed clear improvement when provided with contextual examples of money market returns, moving from incoherent repetition to reasonable numerical predictions. This demonstrates the power of ICL even in smaller models when applied to domain-specific tasks.

The structured format of examples proved effective in guiding the model toward the desired output format, supporting the paper's findings about the importance of example quality and structure in ICL scenarios.

---

## 🚀 Future Directions

This experiment opens several interesting avenues for further exploration:

1. **Dynamic Examples**: Incorporating real-time market data to provide more current examples.
2. **Multi-factor Analysis**: Including additional context like market conditions, economic indicators, or fund characteristics.
3. **Comparative Analysis**: Testing different prompt structures and example quantities to optimize performance.
4. **Evaluation Metrics**: Developing systematic ways to evaluate the accuracy of financial predictions in ICL settings.

---

## 💭 Conclusion

While this experiment shows promising results for using ICL in financial prediction tasks, it's important to note that these predictions should not be used for actual investment decisions. The real value lies in demonstrating how language models can leverage contextual information to generate domain-specific outputs, even with limited training data.

The combination of domain-specific fine-tuning and in-context learning presents an interesting approach for specialized applications where traditional training data might be limited or expensive to obtain.

---

## 🛠️ Acknowledgments

This blog post was enhanced using several AI tools to improve quality and readability:
- **NotebookLM** for generating initial summaries and insights
- **Grammarly** for grammar checking and language refinement
- **ChatGPT** for structuring and organizing the content flow

---

*This experiment was conducted for educational purposes to explore ICL capabilities in financial domains. Always consult qualified financial advisors for investment decisions.*
