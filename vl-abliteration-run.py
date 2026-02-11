import os
import random
import torch

from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

INSTRUCTION_COUNT = 32
LAYER_RATIO = 0.6
POS = -1

HARMFUL_FILE = "harmful.txt"
HARMLESS_FILE = "harmless.txt"

OUTPUT_DIR = "output-model"

def load_model(model_id):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    return model, processor

def read_instruction_files():
    with open(HARMFUL_FILE, "r") as f:
        harmful = f.readlines()

    with open(HARMLESS_FILE, "r") as f:
        harmless = f.readlines()

    return harmful, harmless

def generate_hidden_states(model, processor, instructions, layer_idx):
    hidden_states = []

    for insn in tqdm(instructions, desc="Generating hidden states"):
        inputs = processor(
            text=insn.strip(),
            images=None,
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        h = out.hidden_states[0][layer_idx][:, POS, :]
        hidden_states.append(h)

    return torch.cat(hidden_states, dim=0)

def compute_refusal_direction(model, processor):
    harmful, harmless = read_instruction_files()

    harmful = random.sample(harmful, INSTRUCTION_COUNT)
    harmless = random.sample(harmless, INSTRUCTION_COUNT)

    text_layers = model.model.language_model.model.layers
    total_layers = len(text_layers)
    layer_idx = int(total_layers * LAYER_RATIO)

    print(f"Instruction count: {INSTRUCTION_COUNT}")
    print(f"Total text layers: {total_layers}")
    print(f"Using layer_idx: {layer_idx}")

    harmful_hidden = generate_hidden_states(
        model, processor, harmful, layer_idx
    )
    harmless_hidden = generate_hidden_states(
        model, processor, harmless, layer_idx
    )

    harmful_mean = harmful_hidden.mean(dim=0)
    harmless_mean = harmless_hidden.mean(dim=0)

    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()

    return refusal_dir, layer_idx

def apply_abliteration(model, refusal_dir, layer_idx):
    layer = model.model.language_model.model.layers[layer_idx]

    # Qwen MLP output projection
    W = layer.mlp.down_proj.weight.data

    r = refusal_dir.to(W.device).to(W.dtype)
    r = r / r.norm()

    proj = torch.outer(r, r)
    W -= proj @ W

    return model

def main():
    print("Loading Qwen3-VL model")
    model, processor = load_model(MODEL_ID)

    print("Computing refusal direction")
    refusal_dir, layer_idx = compute_refusal_direction(model, processor)

    refusal_path = MODEL_ID.replace("/", "_") + "_refusal_dir.pt"
    torch.save(refusal_dir, refusal_path)
    print(f"Saved refusal direction to {refusal_path}")

    print("Applying abliteration to text backbone")
    model = apply_abliteration(model, refusal_dir, layer_idx)

    print("Saving abliterated model to output-model/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True
    )
    processor.save_pretrained(OUTPUT_DIR)

    print("Abliterated Qwen3-VL model saved successfully")
    print("Done")

if __name__ == "__main__":
    main()
