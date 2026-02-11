
```
State: Experimental
```

Abliteration pipeline for transformer-based language models, focused on identifying and neutralizing refusal-related behavior at the representation level. It loads Qwen-family or compatible causal LLMs, samples harmful and harmless instructions, and computes a refusal direction by contrasting their mean hidden states at a configurable transformer layer. The app then applies targeted orthogonalization to the model’s MLP projection weights to remove that direction, effectively steering model behavior without full retraining. Designed for reproducibility and experimentation, it supports configurable layer-depth selection, efficient inference-only execution, and clean export of both the computed refusal vector and the modified model for downstream use or sharing.

## Apply Abliteration

- Iterates through each layer.

    - Orthogonalizes weights in:

```
MLP down projection (mlp.down_proj)
Attention output projection (self_attn.o_proj)
```

```py
for layer in layers:
    layer.mlp.down_proj.weight = orthogonalize_matrix(layer.mlp.down_proj.weight, refusal_dir)
    layer.self_attn.o_proj.weight = orthogonalize_matrix(layer.self_attn.o_proj.weight, refusal_dir)
```

References: [sumandora / remove-refusals-with-transformers](https://github.com/sumandora/remove-refusals-with-transformers), [Maxime Labonne – Uncensor Any LLM with Abliteration](https://mlabonne.github.io/blog/posts/2024-06-04_Uncensor_any_LLM_with_abliteration.html)
