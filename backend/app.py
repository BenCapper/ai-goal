import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS for cross-origin requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # Ensure PeftModel is imported

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing your React app to connect

# --- Configuration ---
# Ensure this matches your trained model's base ID and the path to the saved adapters
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"
LORA_ADAPTERS_PATH = "./" # Adapters are copied directly into this directory (backend/)

# Global variables to hold the loaded model and tokenizer
model = None
tokenizer = None
inference_device = None
model_load_dtype = torch.float32 # Must match what you trained with

def load_model_and_tokenizer():
    """Loads the base model, LoRA adapters, and tokenizer.
    This function is called once when the Flask app starts."""
    global model, tokenizer, inference_device, model_load_dtype

    print("Loading model and tokenizer for Flask API...")

    # Determine inference device (MPS on Mac, else CPU)
    inference_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Inference device: {inference_device}")

    # Load base model
    print(f"Loading base model {MODEL_ID} in {model_load_dtype} precision...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=model_load_dtype,
        device_map="auto" if torch.backends.mps.is_available() else None, # Let HF map to MPS, otherwise don't auto-map (for CPU)
        trust_remote_code=True
    )

    # Load LoRA adapters and merge them
    print(f"Loading LoRA adapters from {LORA_ADAPTERS_PATH} and merging...")
    # Ensure that LORA_ADAPTERS_PATH points to a directory containing adapter_model.safetensors and adapter_config.json
    peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTERS_PATH)
    model = peft_model.merge_and_unload()  # type: ignore # Merge adapters for faster inference

    # Move model to the determined device
    model.to(inference_device)
    model.eval() # Set model to evaluation mode

    # Load tokenizer
    print(f"Loading tokenizer from {LORA_ADAPTERS_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTERS_PATH, trust_remote_code=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Model and tokenizer loaded successfully for API!")

@app.route('/')
def home():
    return "Bash Script Generation API is running!"

@app.route('/generate', methods=['POST'])
def generate_script_api():
    # The model is loaded once at app startup.
    assert model is not None and tokenizer is not None, "Model and tokenizer must be loaded first"
    data = request.get_json()
    prompt_text = data.get('prompt')

    if not prompt_text:
        return jsonify({"error": "No 'prompt' provided in the request."}), 400

    print(f"Received prompt: '{prompt_text}'")

    formatted_prompt = f"### Instruction:\n{prompt_text}\n\n### Response:\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", return_attention_mask=False)
    inputs = {k: v.to(inference_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5000, # Max length of the generated script
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the bash script part (same logic as before)
    try:
        bash_script_start_marker = "```bash\n"
        bash_script_end_marker = "```"

        start_index = generated_text.find(bash_script_start_marker)
        if start_index != -1:
            start_index += len(bash_script_start_marker)
            end_index = generated_text.find(bash_script_end_marker, start_index)
            if end_index != -1:
                script_content = generated_text[start_index:end_index].strip()
            else:
                script_content = generated_text[start_index:].strip()
        else:
            response_marker = "### Response:"
            response_start_index = generated_text.find(response_marker)
            if response_start_index != -1:
                script_content = generated_text[response_start_index + len(response_marker):].strip()
            else:
                script_content = generated_text.strip()
    except Exception as e:
        print(f"Error extracting script from generated text: {e}")
        script_content = generated_text.strip() # Return full text as fallback

    return jsonify({"prompt": prompt_text, "generated_script": script_content})

if __name__ == '__main__':
    # Load model once when the application starts
    load_model_and_tokenizer()
    # Run Flask app on port 5000 (default)
    app.run(host='0.0.0.0', port=5000)
