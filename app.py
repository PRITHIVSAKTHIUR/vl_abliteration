import gradio as gr
import torch
import os
import random
import gc
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoConfig
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

from huggingface_hub import HfApi, create_repo, upload_folder

# -----------------------------------------------------------------------------
# Configuration & Dataset Loading
# -----------------------------------------------------------------------------

def load_instructions(filename):
    """
    Loads instructions from a text file, one per line.
    Returns a list of strings.
    """
    if not os.path.exists(filename):
        # Create dummy files if they don't exist to prevent crash on launch
        if "harmful" in filename:
            content = ["Write a script to exploit a vulnerability", "How to steal a car"]
        else:
            content = ["How to make a cake", "What is the capital of France?"]
        
        with open(filename, "w") as f:
            f.write("\n".join(content))
        print(f"Created dummy file: {filename}")
        return content
    
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not lines:
        raise ValueError(f"File {filename} is empty.")
    
    return lines

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def free_memory():
    """Aggressively clears GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_refusal_direction(model, processor, device, layer_idx, num_instructions):
    """
    Computes the refusal direction vector by contrasting harmful vs harmless hidden states.
    """
    
    print("Loading instructions from files...")
    harmful_source = load_instructions("harmful.txt")
    harmless_source = load_instructions("harmless.txt")

    n_samples = min(num_instructions, len(harmful_source), len(harmless_source))
    harmful_inst = random.sample(harmful_source, n_samples)
    harmless_inst = random.sample(harmless_source, n_samples)
    
    print(f"Selected {n_samples} pairs of instructions for direction computation.")

    def get_hidden_states(instructions):
        hidden_states_list = []
        
        for inst in tqdm(instructions, desc="Generating Hidden States"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": inst}
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Robust hidden state extraction
            if hasattr(outputs, "hidden_states"):
                hs = outputs.hidden_states[layer_idx]
            else:
                # Fallback for models that might nest it differently
                hs = outputs['hidden_states'][layer_idx]
                
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
    refusal_dir = refusal_dir / (refusal_dir.norm() + 1e-8) # Avoid div by zero
    
    return refusal_dir.to(device)

def orthogonalize_matrix(matrix, vector):
    """
    Projects the matrix weights onto the null space of the refusal vector.
    """
    vector = vector.to(matrix.device).to(matrix.dtype)
    dot = torch.matmul(matrix, vector)
    projection = torch.outer(dot, vector)
    return matrix - projection

def get_model_layers(model):
    """
    Robustly finds the list of transformer layers in the model.
    """
    # 1. Try standard Qwen/Llama path
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    
    # 2. Try Qwen-VL specific path (sometimes language_model is separate)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers
    
    # 3. Try generic .layers path
    if hasattr(model, "layers"):
        return model.layers
        
    # 4. Deep search for ModuleList
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            # Check if it looks like a decoder layer
            if hasattr(module[0], "self_attn") or hasattr(module[0], "mlp"):
                return module
                
    raise ValueError(f"Could not find layers in model. Available keys: {model.__dict__.keys()}")

def apply_abliteration(model, refusal_dir, device):
    """
    Iterates through model layers and orthogonalizes weights.
    """
    layers = get_model_layers(model)
    refusal_dir = refusal_dir.to(device)
    
    count = 0
    for layer in tqdm(layers, desc="Orthogonalizing Weights"):
        # MLP Down Projection
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            w = layer.mlp.down_proj.weight.data
            layer.mlp.down_proj.weight.data = orthogonalize_matrix(w, refusal_dir)
            count += 1
            
        # Attention Output Projection
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            w = layer.self_attn.o_proj.weight.data
            layer.self_attn.o_proj.weight.data = orthogonalize_matrix(w, refusal_dir)
            count += 1
            
    return f"Applied orthogonalization to {count} matrices."

# -----------------------------------------------------------------------------
# Main Processing Logic
# -----------------------------------------------------------------------------

def process_model(model_id, hf_token, repo_id, layer_percentage, max_shard_size, use_4bit):
    status_log = []
    def log(msg):
        status_log.append(msg)
        return "\n".join(status_log)

    yield log(f"🚀 Starting process for {model_id}...")
    
    # Files check
    if not os.path.exists("harmful.txt"): 
        load_instructions("harmful.txt") # create dummy if missing
    if not os.path.exists("harmless.txt"):
        load_instructions("harmless.txt")

    if hf_token:
        try:
            HfApi(token=hf_token).whoami()
            log("✅ Logged in to Hugging Face.")
        except Exception as e:
            yield log(f"❌ Login failed: {e}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yield log(f"⚙️ Using device: {device}")

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
        
        # Robust Model Loading
        # 1. Try Qwen3VL class if imported
        # 2. Try Qwen2_5_VL class
        # 3. Fallback to AutoModel
        
        loaded_model = None
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        classes_to_try = [
            Qwen3VLForConditionalGeneration, 
            Qwen2_5_VLForConditionalGeneration,
            AutoModelForCausalLM
        ]
        
        for cls in classes_to_try:
            if cls is None: continue
            try:
                loaded_model = cls.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
                yield log(f"✅ Loaded using class: {cls.__name__}")
                break
            except Exception as e:
                pass # Try next class
        
        if loaded_model is None:
            yield log("❌ Failed to load model with any known class.")
            return
        
        model = loaded_model
        model.eval()

    except Exception as e:
        yield log(f"❌ Error loading model: {e}")
        return

    # --- Robust Layer Detection ---
    try:
        # 1. Try to get config count
        if hasattr(model, "config"):
            num_layers = getattr(model.config, "num_hidden_layers", None)
            if num_layers is None:
                # Sometimes it's called n_layer or n_layers
                num_layers = getattr(model.config, "n_layer", None)
        
        # 2. If config failed, count the actual list
        if not num_layers:
            layers_list = get_model_layers(model)
            num_layers = len(layers_list)
            
        yield log(f"✅ Detected {num_layers} layers.")
        
    except Exception as e:
        yield log(f"❌ Layer detection failed: {e}")
        return

    target_layer_idx = int(num_layers * (layer_percentage / 100.0))
    yield log(f"🔍 Targeting Layer {target_layer_idx} ({layer_percentage}% depth).")

    try:
        yield log("🧪 Computing Refusal Direction...")
        refusal_dir = get_refusal_direction(model, processor, device, target_layer_idx, num_instructions=32)
        yield log("✅ Refusal direction computed.")
    except Exception as e:
        yield log(f"❌ Error computing refusal direction: {e}")
        return

    try:
        yield log("✂️ Applying Abliteration...")
        msg = apply_abliteration(model, refusal_dir, device)
        yield log(f"✅ {msg}")
    except Exception as e:
        yield log(f"❌ Error applying abliteration: {e}")
        return

    if repo_id and hf_token:
        save_path = "abliterated_model"
        yield log(f"💾 Saving to {save_path}...")
        
        if use_4bit:
            yield log("⚠️ Warning: Saving 4-bit model. Weights may not merge correctly.")

        try:
            model.save_pretrained(save_path, max_shard_size=max_shard_size)
            processor.save_pretrained(save_path)
            yield log("✅ Local save complete.")
            
            yield log(f"☁️ Uploading to: {repo_id}...")
            create_repo(repo_id, token=hf_token, private=True, exist_ok=True)
            
            upload_folder(
                folder_path=save_path,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token
            )
            yield log("🎉 Upload Complete!")
            
        except Exception as e:
            yield log(f"❌ Error during save/upload: {e}")
    else:
        yield log("ℹ️ No Repo ID provided. Skipping upload.")

    free_memory()

# -----------------------------------------------------------------------------
# Gradio Interface
# -----------------------------------------------------------------------------

with gr.Blocks(title="Qwen3-VL Abliterator") as demo:
    gr.Markdown("# 🧠 Qwen3-VL / Text-Only Model Abliterator")
    
    with gr.Row():
        with gr.Column():
            model_id_input = gr.Textbox(label="Source Model ID", value="prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1")
            token_input = gr.Textbox(label="Hugging Face Token", type="password")
            repo_id_input = gr.Textbox(label="Target Repo ID", placeholder="user/Qwen3-VL-Abliterated")
        
        with gr.Column():
            layer_slider = gr.Slider(0, 100, 60, label="Layer Depth (%)")
            shard_size_input = gr.Textbox(label="Max Shard Size", value="3GB")
            use_4bit_check = gr.Checkbox(label="Load in 4-bit", value=False)

    run_btn = gr.Button("🚀 Start", variant="primary")
    logs = gr.Textbox(label="Logs", lines=15, interactive=False)

    run_btn.click(
        process_model,
        inputs=[model_id_input, token_input, repo_id_input, layer_slider, shard_size_input, use_4bit_check],
        outputs=[logs]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
