import gradio as gr
import torch
import os
import random
import gc
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoConfig
)
from huggingface_hub import HfApi, create_repo, upload_folder

# -----------------------------------------------------------------------------
# Import Qwen3-VL Specific Classes
# -----------------------------------------------------------------------------
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def free_memory():
    """Aggressively clears GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_instructions(filename):
    """Loads instructions from a text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Required file '{filename}' not found in the directory.")
    
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not lines:
        raise ValueError(f"File '{filename}' is empty.")
        
    return lines

def get_model_layers(model):
    """
    Robustly finds the list of transformer layers in Qwen/Qwen-VL models.
    """
    # 1. Standard Qwen / Llama structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    
    # 2. Qwen-VL specific (often wrapped in language_model)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers
    
    # 3. Generic fallback
    if hasattr(model, "layers"):
        return model.layers
        
    # 4. Deep search for ModuleList
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            if hasattr(module[0], "self_attn") or hasattr(module[0], "mlp") or hasattr(module[0], "attn"):
                return module
                
    raise ValueError(f"Could not find layers in model. Available keys: {model.__dict__.keys()}")

def get_refusal_direction(model, processor, device, layer_idx, num_instructions):
    """
    Computes the refusal direction vector by contrasting harmful vs harmless hidden states.
    """
    print("Loading instructions from local files...")
    try:
        harmful_source = load_instructions("harmful.txt")
        harmless_source = load_instructions("harmless.txt")
    except Exception as e:
        raise e

    n_samples = min(num_instructions, len(harmful_source), len(harmless_source))
    harmful_inst = random.sample(harmful_source, n_samples)
    harmless_inst = random.sample(harmless_source, n_samples)
    
    print(f"Selected {n_samples} pairs of instructions.")

    def get_hidden_states(instructions):
        hidden_states_list = []
        
        for inst in tqdm(instructions, desc="Generating Hidden States"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": inst}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
            
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Extract hidden states (usually layer_idx + 1 due to embeddings)
            if hasattr(outputs, "hidden_states"):
                hs = outputs.hidden_states[layer_idx + 1]
            else:
                hs = outputs['hidden_states'][layer_idx + 1]
                
            last_token_hs = hs[:, -1, :] 
            hidden_states_list.append(last_token_hs.cpu())
            
        return torch.cat(hidden_states_list, dim=0)

    print("--- Computing Harmful Hidden States ---")
    harmful_h = get_hidden_states(harmful_inst)
    
    print("--- Computing Harmless Hidden States ---")
    harmless_h = get_hidden_states(harmless_inst)
    
    harmful_mean = harmful_h.mean(dim=0)
    harmless_mean = harmless_h.mean(dim=0)
    
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / (refusal_dir.norm() + 1e-8)
    
    return refusal_dir.to(device)

def apply_abliteration(model, refusal_dir, device):
    """
    Robustly iterates through model layers and orthogonalizes weights.
    Uses weight SHAPE detection to find target matrices regardless of naming.
    """
    layers = get_model_layers(model)
    
    refusal_dir = refusal_dir.to(device)
    refusal_dir = refusal_dir.to(dtype=model.dtype) 
    
    # hidden_size is the length of the refusal vector
    hidden_size = refusal_dir.shape[-1]
    
    count = 0
    modified_types = []

    print(f"Scanning for Linear layers with Output Dimension: {hidden_size}...")

    # Names to strictly avoid (Input projections)
    # We only want to abliterate layers that WRITE to the residual stream (Output/Down projections)
    avoid_substrings = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "w1", "w3"]

    for i, layer in enumerate(tqdm(layers, desc="Scanning Layers")):
        # Recursive search for all modules in this layer
        for name, module in layer.named_modules():
            
            # Check if it has a weight attribute (covers Linear, Linear4bit, etc.)
            if hasattr(module, "weight"):
                w = module.weight
                
                # Check dimensions
                # PyTorch Linear weights: [out_features, in_features]
                # We want out_features == hidden_size
                if w.shape[0] == hidden_size:
                    
                    # Double check it's not a prohibited layer type (like Query/Key/Value)
                    if any(bad in name for bad in avoid_substrings):
                        continue
                        
                    # If we passed checks, this is likely an output projection (down_proj or o_proj)
                    try:
                        # Normalize Refusal Vector
                        v = refusal_dir.view(-1)
                        v_norm = v / (v.norm() + 1e-8)
                        
                        # --- Orthogonalization Math ---
                        # We want to remove v from the output space (columns of W^T, or rows of W).
                        # Matrix Correction = OuterProduct(v, v @ W)
                        # v [d] . W [d, k] -> overlap [k]
                        # v [d] * overlap [k] -> correction [d, k]
                        
                        # Note on shapes:
                        # v_norm is [hidden_size]
                        # w.data is [hidden_size, in_features]
                        
                        # Calculate overlap of vector with every column of weight matrix?
                        # No, we assume W maps TO hidden_size.
                        # We project the rows of W onto the null space of v.
                        
                        overlap = torch.matmul(v_norm, w.data) # Result: [in_features]
                        correction = torch.outer(v_norm, overlap) # Result: [hidden_size, in_features]
                        
                        # Apply subtraction
                        module.weight.data -= correction
                        count += 1
                        
                        short_name = name.split(".")[-1]
                        if short_name not in modified_types:
                            modified_types.append(short_name)
                            
                    except Exception as e:
                        print(f"⚠️ Failed to modify {name}: {e}")

    if count == 0:
        return f"❌ Error: Found 0 matrices with output size {hidden_size}. Model structure mismatch."
        
    return f"✅ Success: Modified {count} matrices. (Found types: {modified_types})"

# -----------------------------------------------------------------------------
# Main Processing Generator
# -----------------------------------------------------------------------------

def process_model(model_id, hf_token, repo_id, layer_percentage, max_shard_size, use_4bit):
    status_log = []
    def log(msg):
        status_log.append(msg)
        return "\n".join(status_log)

    yield log(f"🚀 Starting process for {model_id}...")
    
    # 1. Check Files
    if not os.path.exists("harmful.txt") or not os.path.exists("harmless.txt"):
        yield log("❌ Error: 'harmful.txt' or 'harmless.txt' not found in current directory.")
        return

    # 2. Auth
    if hf_token:
        try:
            HfApi(token=hf_token).whoami()
            log("✅ Logged in to Hugging Face.")
        except Exception as e:
            yield log(f"❌ Login failed: {e}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yield log(f"⚙️ Using device: {device}")

    # 3. Load Model
    try:
        yield log("📥 Loading model...")
        
        quant_config = None
        if use_4bit and device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        
        # Priority Loading Logic
        model_cls = AutoModelForCausalLM # Default fallback
        
        if Qwen3VLForConditionalGeneration is not None:
            yield log("ℹ️ Class found: Qwen3VLForConditionalGeneration. Using it.")
            model_cls = Qwen3VLForConditionalGeneration
        elif Qwen2_5_VLForConditionalGeneration is not None and "2.5" in model_id:
            yield log("ℹ️ Class found: Qwen2_5_VLForConditionalGeneration. Using it.")
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            yield log("⚠️ Qwen3/2.5 classes not found or applicable. Using AutoModelForCausalLM.")

        model = model_cls.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        model.eval()
        yield log(f"✅ Loaded {model_id}.")

    except Exception as e:
        yield log(f"❌ Error loading model: {e}")
        return

    # 4. Detect Layers
    try:
        layers_list = get_model_layers(model)
        num_layers = len(layers_list)
        yield log(f"✅ Detected {num_layers} layers.")
    except Exception as e:
        yield log(f"❌ Layer detection failed: {e}")
        return

    target_layer_idx = int(num_layers * (layer_percentage / 100.0))
    target_layer_idx = max(0, min(target_layer_idx, num_layers - 1))
    
    yield log(f"🔍 Targeting Layer {target_layer_idx} ({layer_percentage}% depth).")

    # 5. Compute Vector
    try:
        yield log("🧪 Computing Refusal Direction from local files...")
        refusal_dir = get_refusal_direction(model, processor, device, target_layer_idx, num_instructions=32)
        yield log("✅ Refusal direction computed.")
    except Exception as e:
        yield log(f"❌ Error computing refusal direction: {e}")
        return

    # 6. Apply Abliteration
    try:
        yield log("✂️ Applying Abliteration...")
        msg = apply_abliteration(model, refusal_dir, device)
        yield log(msg)
        if "❌" in msg: return # Stop if abliteration failed
    except Exception as e:
        yield log(f"❌ Error applying abliteration: {e}")
        return

    # 7. Save & Upload
    if repo_id and hf_token:
        save_path = "abliterated_model"
        yield log(f"💾 Saving to '{save_path}'...")
        
        if use_4bit:
            yield log("⚠️ Warning: Saving 4-bit loaded model. Weights might not save in FP16/BF16 correctly.")

        try:
            # Save
            model.save_pretrained(save_path, max_shard_size=max_shard_size)
            processor.save_pretrained(save_path)
            yield log("✅ Local save complete.")
            
            # Upload
            yield log(f"☁️ Uploading to: {repo_id}...")
            create_repo(repo_id, token=hf_token, private=True, exist_ok=True)
            
            upload_folder(
                folder_path=save_path,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token
            )
            yield log(f"🎉 Upload Complete! Model at: https://huggingface.co/{repo_id}")
            
        except Exception as e:
            yield log(f"❌ Error during save/upload: {e}")
    else:
        yield log("ℹ️ No Repo ID provided. Skipping upload.")

    free_memory()
    yield log("🏁 Process Finished.")

# -----------------------------------------------------------------------------
# Gradio Interface
# -----------------------------------------------------------------------------

with gr.Blocks(title="Qwen3-VL Abliterator") as demo:
    gr.Markdown("# 🧠 Qwen3-VL / Qwen2.5-VL Abliterator (One-Click)")
    gr.Markdown("Reads `harmful.txt` and `harmless.txt` from the current directory, calculates the refusal vector, removes it, and uploads the model.")
    
    with gr.Row():
        with gr.Column():
            model_id_input = gr.Textbox(label="Source Model ID", value="Qwen/Qwen2.5-VL-7B-Instruct")
            token_input = gr.Textbox(label="Hugging Face Write Token", type="password")
            repo_id_input = gr.Textbox(label="Target Repo ID", placeholder="user/Qwen2.5-VL-Abliterated")
        
        with gr.Column():
            layer_slider = gr.Slider(0, 100, 60, label="Layer Depth (%) - 60% Recommended")
            shard_size_input = gr.Textbox(label="Max Shard Size", value="3GB")
            use_4bit_check = gr.Checkbox(label="Load in 4-bit (Saves VRAM, risks save issues)", value=False)

    run_btn = gr.Button("🚀 Start Full Process", variant="primary")
    logs = gr.Textbox(label="Process Logs", lines=20, interactive=False)

    run_btn.click(
        process_model,
        inputs=[model_id_input, token_input, repo_id_input, layer_slider, shard_size_input, use_4bit_check],
        outputs=[logs]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
