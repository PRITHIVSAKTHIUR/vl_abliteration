vl_abliteration is a GUI-based tool designed to abliterat(e) vision-language and text-only transformer models in a controlled, research-friendly way. Built with Gradio, it lets users load Qwen-family VL models or standard causal LLMs, compute a refusal direction by contrasting harmful and harmless instructions, and then neutralize that direction by orthogonalizing key attention and MLP projection weights across selected layers. The app supports flexible layer-depth targeting, optional 4-bit loading for memory efficiency, and end-to-end workflows including local saving and direct upload to Hugging Face. With robust model and layer detection, automatic processor handling, and real-time logs, vl_abliteration makes experimenting with safety, alignment, and behavior steering in multimodal transformer models accessible through a clean, interactive interface.

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

```
State: Experimental
```
