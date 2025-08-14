# Measuring Model Fluency with Perplexity — A Hands-On Practice

*August 14, 2025*

In my continuing journey to understand LLM evaluation, I explored perplexity, a classic metric introduced by Jelinek et al. in "Perplexity—a measure of the difficulty of speech recognition tasks" (1977) ([link](https://watermark.silverchair.com/s63_5_online.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAACAowgggGBgkqhkiG9w0BBwagggf3MIIH8wIBADCCB-wGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMu1YoCsB-egNhbZ2mAgEQgIIHvYn0yiFwFwe5m0ceuiY8pHAuJPT0V96Mu8m9TeGVBobJxx1n8c7HQmPCnzuguvcDkOBVyOzoqVeX7QipVd_LwIV9zsFObhLKWuElqU_n_Nf8Rq1YdgsLVf8J_iS8AnbMNg4-jOO5_z2Qi853vYC-VHC0UIyJK-OU2riqdDOvOby0adwF2vXCx5ZHJRVvvJuDmzPU_uT_iHuINAgbnWlMDtS4XboomMpim5WlqXSuO8bNHHm-HTWayLVLGCsM3lBZk3zKEcXpgsDVF2KuIl-YYpQejnHSO1ggKSOeoi8MuECK8sQj-HnNSlYXWfpkZZ01GNnKVHuqJg2dyzb4t97czLRI9uyVXLuV0huFz9gUJ-9fUU91PouVpeeM1_DLf3dOJXSWFh1BipOsI_yLf8-Boxa7e5--CIjearH2cONI9rnHFW0dN7sJS9lkvIVvtbF5_agmNnCaRmTp7CeCoNTrAPJ8hbBOJuwDqxP7ZY0KsvuEgZz7dhkpJ9v27CvbwZON0LO254HpLDTK6tk39pXGAGLSunhVd-JKmdSR6QQOzTOk413qhKO3kvbyonCdrtdATow2_h_jroIDnOnertLcqv62wCYq5akxyDIqNqtU85ozBENpshPHF3ZZ4xEreSouj-KszoTjfdnID3SRsFpH3ykNgHwLzBfHh71CHmoi4OMWf7KAWrcOud3s03_dsLmBQp646bMaJQX3EE80m7pBiZT6LudTsE8F8BkenAterxLnd2oD13ql7pfmI76F9aHib_G4AfLCTHS_Jfyq0OmO5X6qPXnOFRedy_6HkwLfBzq_IrKnrExnjl4dPQZyEDA0wJlHfCUzVxKGVgOR1_MiCiug-bfVNvZLfV4r542QSyI19wHse3_UeSlO4sjESI-fkVLv3_c3_sCAJ6tZBGj-VxpZ_ss1xmnYmswG8WwltUwISPQbb3LATKQmuUy-FyCLxLxKpNq5tlb_JNaqpU2lh6Exa8lTYz14-c3Qhzjsdx6BOW2x6gLfcf1Sz09eJ2lgXKGkSGDosDlKNGsNdkS6d_d4SdfykXICRV89w8zENiQqRqTwzXVqwioR6ZVvYCj7l430x0U2vULQgdyi_N2e356Qi90lKI-6JpD8bZMeGAf-T7QSClV2OyW9aIeWKfRe-rBrH26KNed-JufXwza2QPipziW3wIcVFXak5MS14mBJqu6zpv9xa8ckD8Twi8cDf1wG4GzHzds5hOFNAGHvHRrkC7qCR-upBcGMVjlUCpuyu7voZgMnIJif9DrXR1ViMeMJRMQE765rdAfcrA8uLV_n6mV5bzowAWheYUQJwONEOHotkWRXsxrbTn4KfMMNHlqL-bIp1Q121ESBTUR9l6TI1sOV10WwZl5XE3q927twUyOgzhUgZHSYX5iwyunBwqO4p3kNTdaPOCAH60hbE8Fa5juQkVdLtzertcR28x4F4ucFy4gMTPzHgDo92DnK-R0nUJlgCIZzeZqda-5Hmvs0wCp9jqT9a0QFsL1qBvH_kuIAMwZ_enW0RMPxMJel3UBpzt-_u0bUVKtEbVt390de4cZwytU0DFz8GChPRK-ay69E2xjaEahxch6eJWbcmFLei7whUlxyMpKZbHY3LW-lDohqWMDnYt4jc7AGqdpgWjZBjmCKgDTOQR5UUEymq14dEyqrt-INdopTrGsirLSvdG8fF5a2XjMALt-SBclSxMrxYighBFk7WSOkpBnmUFL4Dr1kp35rUezenkUaW1yYUe7fntui8omFCB-XrfAWen_tGIT-kaSvDOBpvtU5uPDxqkmVr2rGfqGCgxd9qnuSAjDs6h0zL9cGgxte8aaDweg-xpsypRyNk9QTkm4zdOV5UrS7YCkQH599Z2d3Ip5RFxX3dN1r1kmihsSUXrmjrovS5y4EnRyRrNLVTVCvVrvo5P0uGXBSKM-vZ4cRJvDj7McygMIVGs3Y1ulGEf-cvHXpbBhVRR1ZUKNuzbiDYvTUPvyl-Zwk3JKDC_yB4_wzBdCxdN9q4fcJis5uAVHcWLG8StEoNv4KRMIsvCan4zyLXGu22ctEiIZhXqyQ-c3hQ_cCXL9lUKw1lQrFyudChjl-35fIJGJV9kZ1VxelAdKQ5yc0RwMivtW8qSQNT7kN-di69CPso2iMq310nYXnRsF9zhzXs3PPX6AjIaq-EXgbZFcWK-PlQ9EWagAL1PtYryikvvfOPk62GNEtDL9wi6Z7NxjTgpr-TLXPbf7bfNaQVVlc4ztT7grdXtwI8UTw058caXxmGi2YRR-u6ZBVJK1hiQhpvt7PNX8hl2Opva1gCkDXEm4Q26lxRdt2ztTVqe4mn4anTAKhkcOwbkPWyLSXRVS9xwUlfT-aulk39Na5ufz3420FNwMF64cWeK8rNyerYD-K07VOG9RgUHkGcla_gDmyZuEUIysV4w6-4rW36EePKXPkqzwqlqYaXtxU8XkGU9a50RW9GsvdX5IVHsgZ7hThk1FkVPQfZ70PlAMD7WIiMzfrshWWyV2t-XDuODgWIQ5OVCxWO-oyHP12s9UvsOpdJ-1_xLIL7XMqNAC_2g40gK9x8ONu5fMmEmng_7n__mlEA2BKRT8ZwtNu2ezJ9yAPdZ67_DvHALtvFYy3h0gvQo3ToCWG8WQ)). The concept is simple but powerful: perplexity measures how well a language model predicts the next word in a sequence. Lower perplexity means the model is more confident and accurate in its predictions.

## Understanding Perplexity

The original paper by Jelinek and colleagues established perplexity as a fundamental metric for evaluating language models. It's mathematically defined as the exponential of the cross-entropy loss, representing how "perplexed" or uncertain a model is when predicting the next token in a sequence. The beauty of perplexity lies in its simplicity: it directly reflects how well a model has learned the statistical patterns of language.

## My Implementation

For this exercise, I used my fine-tuned Wekeza GPT-2 model and computed perplexity for short finance-related sentences. The notebook loads the model, tokenizes sample texts, runs them through the model in evaluation mode, and calculates perplexity from the loss.

Here's the setup code:

```python
!pip install transformers datasets torch --quiet

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
import math

model_path = "./distilgpt2-wekeza-finetuned_v5_cot_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.eval()
```

I created a simple dataset with finance-related sentences:

```python
data = [
    "Money market funds offer liquidity with low risk.",
    "Fixed deposits lock funds for higher interest rates."
]
dataset = Dataset.from_list([{"text": t} for t in data])
```

The perplexity calculation function is straightforward:

```python
def compute_ppl(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())

for example in dataset:
    ppl = compute_ppl(example["text"])
    print(f"Text: {example['text']}\nPerplexity: {ppl:.2f}\n")
```

## Results and Insights

Results were telling:

"Money market funds offer liquidity with low risk." — Perplexity ~70

"Fixed deposits lock funds for higher interest rates." — Perplexity ~185

These scores highlight variations in model familiarity with specific phrasing and domain patterns. The first sentence received a much lower perplexity score, suggesting my model finds this phrasing more predictable, likely due to similar patterns in the training data. The higher perplexity for the second sentence indicates the model is less certain about this particular word sequence.

This aligns with the paper's core idea that perplexity reflects the predictability of text under a given model. The metric essentially quantifies how surprised the model is by each word in the sequence, averaged across the entire text.

## Technical Observations

During implementation, I noticed the familiar LoRA adapter warnings and the automatic loss type configuration. The model ran smoothly on CPU for this small-scale evaluation. The `math.exp()` function converts the raw cross-entropy loss into the more interpretable perplexity score.

One interesting aspect is how perplexity provides a different perspective compared to the generation-based metrics I've been exploring. While ROUGE, BLEU, METEOR, and BERTScore compare generated text against references, perplexity measures the model's internal confidence about a given text sequence.

## Why This Matters

While perplexity doesn't capture semantic accuracy or factuality, it remains a foundational metric in language modeling. It's particularly useful for comparing different models on the same text or tracking how well a model fits to specific domains during training. Lower perplexity generally correlates with better language modeling capability, though it doesn't guarantee better performance on downstream tasks.

Running this hands-on experiment gave me a much better feel for how perplexity works in practice. The metric provides insight into what the model has learned about language patterns, even if it doesn't tell us whether the model generates helpful or accurate responses.

## Continuing the Evaluation Journey

This was another small but meaningful step in my LLM evaluation learning series. Having now explored reference-based metrics (ROUGE-L, BLEU, METEOR, BERTScore) and this reference-free metric (perplexity), I'm building a more complete picture of the evaluation landscape.

Each metric teaches something different about model behavior. Perplexity reveals the model's internal confidence and language modeling quality, while the reference-based metrics assess how well generated text matches expected outputs.

## Resources

The complete code for this experiment is available in my [GitHub repository](https://github.com/Okoth67/llm_eval_perplexity_wekeza/tree/main). The original paper by Jelinek et al. established the theoretical foundation for using perplexity in language model evaluation.

## Acknowledgments

Thanks to Claude (Anthropic) for help with blog structure and organizing these evaluation concepts clearly.

---
