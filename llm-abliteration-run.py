import os
import random
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.set_grad_enabled(False)

# ---------------- CONFIG ----------------

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
INSTRUCTION_COUNT = 32
LAYER_RATIO = 0.6
POS = -1

HARMFUL_FILE = "harmful.txt"
HARMLESS_FILE = "harmless.txt"

OUTPUT_DIR = "output-model"


# ---------------- MODEL LOADING ----------------

def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------- DATA ----------------

def read_instruction_files():
    with open(HARMFUL_FILE, "r") as f:
        harmful = f.readlines()

    with open(HARMLESS_FILE, "r") as f:
        harmless = f.readlines()

    return harmful, harmless


# ---------------- CORE LOGIC ----------------

def generate_hidden_states(model, tokenizer, instructions, layer_idx):
    hidden_states = []

    for insn in tqdm(instructions, desc="Generating hidden states"):
        enc = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": insn}],
            add_generation_prompt=True,
            return_tensors="pt"
        )

        attention_mask = torch.ones_like(enc)

        out = model.generate(
            enc.to(model.device),
            attention_mask=attention_mask.to(model.device),
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        h = out.hidden_states[0][layer_idx][:, POS, :]
        hidden_states.append(h)

    return torch.stack(hidden_states)


def compute_refusal_direction(model, tokenizer):
    harmful, harmless = read_instruction_files()

    harmful = random.sample(harmful, INSTRUCTION_COUNT)
    harmless = random.sample(harmless, INSTRUCTION_COUNT)

    total_layers = len(model.model.layers)
    layer_idx = int(total_layers * LAYER_RATIO)

    print(f"Total layers: {total_layers}")
    print(f"Layer ratio: {LAYER_RATIO}")
    print(f"Using layer_idx: {layer_idx}")

    harmful_hidden = generate_hidden_states(
        model, tokenizer, harmful, layer_idx
    )
    harmless_hidden = generate_hidden_states(
        model, tokenizer, harmless, layer_idx
    )

    harmful_mean = harmful_hidden.mean(dim=0)
    harmless_mean = harmless_hidden.mean(dim=0)

    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()

    return refusal_dir.squeeze(), layer_idx


def apply_abliteration(model, refusal_dir, layer_idx):
    layer = model.model.layers[layer_idx]

    W = layer.mlp.down_proj.weight.data
    r = refusal_dir.to(W.device)

    proj = torch.outer(r, r)
    W -= proj @ W

    return model


# ---------------- MAIN PIPELINE ----------------

def main():
    print("Loading model")
    model, tokenizer = load_model(MODEL_ID)

    print("Computing refusal direction")
    refusal_dir, layer_idx = compute_refusal_direction(
        model, tokenizer
    )

    refusal_path = MODEL_ID.replace("/", "_") + "_refusal_dir.pt"
    torch.save(refusal_dir, refusal_path)
    print(f"Saved refusal direction to {refusal_path}")

    print("Applying abliteration")
    model = apply_abliteration(model, refusal_dir, layer_idx)

    print("Saving abliterated model to output-model/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Abliterated model saved successfully")
    print("Done")


if __name__ == "__main__":
    main()
