import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, DatasetDict, Dataset
from tqdm.auto import tqdm # For progress bars
from accelerate import Accelerator # For simpler device management and mixed precision if applicable

# --- Configuration ---
DATASET_DIR = "." # Directory containing your .jsonl files
TRAIN_FILE = os.path.join(DATASET_DIR, "bash_scripts_dataset_train.jsonl")
VAL_FILE = os.path.join(DATASET_DIR, "bash_scripts_dataset_val.jsonl")

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"
OUTPUT_DIR = "./deepseek_bash_finetuned_adapters_custom" # Base output directory for saved adapters

LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3 # Maximum epochs to run, early stopping will stop it sooner
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
MAX_SEQ_LENGTH = 512 # Max input sequence length

# --- Overfitting Control Parameters ---
EARLY_STOPPING_PATIENCE = 3 # Stop if validation loss doesn't improve for this many epochs
WEIGHT_DECAY = 0.01         # Strength of L2 regularization for the optimizer
LORA_DROPOUT = 0.1         # Dropout for LoRA layers (can increase to 0.1 or 0.15 if needed)

# --- 1. Load Dataset ---
print("Loading dataset...")
try:
    dataset = load_dataset(
        'json',
        data_files={'train': TRAIN_FILE, 'validation': VAL_FILE},
        split='train'
    )
    dataset_dict = DatasetDict({
        'train': dataset,
        'validation': load_dataset('json', data_files=VAL_FILE, split='train')
    })
    print(f"Dataset loaded successfully. Training examples: {len(dataset_dict['train'])}, Validation examples: {len(dataset_dict['validation'])}")
    print(f"Example from training dataset (raw text):\n{dataset_dict['train'][0]['text']}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Please ensure '{TRAIN_FILE}' and '{VAL_FILE}' exist and are valid JSONL files.")
    exit(1)

# --- 2. Load Model & Tokenizer ---
print("\nLoading model directly in float32 precision...")

# Explicitly set model_load_dtype to torch.float32 for training stability on MPS
model_load_dtype = torch.float32

# Set the device for model loading and training
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training device: {device}")

# Check if MPS is available
if not torch.backends.mps.is_available():
    print("WARNING: MPS backend not available. Training will default to CPU, which will be much slower.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=model_load_dtype, # Load model weights in float32
    device_map="auto" if torch.backends.mps.is_available() else None # Let transformers map to MPS, otherwise don't auto-map (for CPU)
)
model.config.use_cache = False

print(f"Model loaded in {model_load_dtype} precision.")
print(f"Model device: {model.device}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.eos_token is None:
    tokenizer.eos_token = "</s>"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Use EOS as pad token for causal LMs

tokenizer.padding_side = "right" # Important for causal LMs
print("Tokenizer loaded.")

# --- 3. Tokenize Dataset (using the tokenizer directly) ---
def tokenize_function(examples):
    # This will directly convert your "text" field into input_ids, attention_mask
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length", # Pad to max_length for batching
        return_tensors="pt" # Return PyTorch tensors
    )

print("\nTokenizing dataset...")
tokenized_dataset_train = dataset_dict['train'].map(
    tokenize_function,
    batched=True,
    remove_columns=['text'], # Remove original text column after tokenization
)
tokenized_dataset_val = dataset_dict['validation'].map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
)

# Convert to PyTorch format (though return_tensors="pt" usually handles this)
tokenized_dataset_train.set_format("torch")
tokenized_dataset_val.set_format("torch")

print("Dataset tokenization complete.")
print(f"Example from tokenized training dataset (input_ids):\n{tokenized_dataset_train[0]['input_ids']}")


# --- 4. Prepare Model for LoRA Fine-tuning ---
print("\nPreparing model for LoRA fine-tuning...")
# prepare_model_for_kbit_training is still useful for memory optimization (e.g., gradient checkpointing)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear", # Target all linear layers for LoRA
    lora_dropout=LORA_DROPOUT,   # Use configurable LORA_DROPOUT
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
print("Model ready for LoRA fine-tuning.")

# --- 5. Custom PyTorch Training Loop ---
print("\nStarting custom PyTorch training loop...")

accelerator = Accelerator(cpu=not torch.backends.mps.is_available())

# Prepare model for training with Accelerator (moves to device and wraps for distributed training if applicable)
model = accelerator.prepare(model)

# Create DataLoaders
train_dataloader = DataLoader(tokenized_dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(tokenized_dataset_val, batch_size=TRAIN_BATCH_SIZE) # No shuffle for validation

# Optimizer (added weight_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Prepare optimizer and dataloaders with Accelerator
optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    optimizer, train_dataloader, val_dataloader
)

# Early Stopping variables
best_val_loss = float('inf')
epochs_without_improvement = 0
best_epoch = -1 # Track which epoch had the best validation loss

# Create the base output directory for all saves
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Create a dedicated directory for saving the best model
os.makedirs(os.path.join(OUTPUT_DIR, "best_model"), exist_ok=True) 

# Training loop
model.train() # Set model to training mode

for epoch in range(NUM_TRAIN_EPOCHS):
    total_loss = 0
    # Use tqdm for a nice progress bar
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_TRAIN_EPOCHS}")):
        
        # Forward pass
        outputs = model(**batch, labels=batch["input_ids"]) # labels are input_ids for causal LM
        loss = outputs.loss

        # Gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        accelerator.backward(loss) # Perform backward pass with Accelerator

        # Optimizer step
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            accelerator.clip_grad_norm_(model.parameters(), 0.3) # Clip gradients for stability
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.detach().float() * GRADIENT_ACCUMULATION_STEPS # Scale back loss for reporting
            
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval() # Set model to evaluation mode
    val_loss = 0
    for step, batch in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")):
        with torch.no_grad():
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
        val_loss += loss.detach().float()
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # --- End-of-Epoch Saving (Robust Checkpoint) ---
    # Save current epoch's model to its own directory, regardless of improvement
    current_epoch_save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    os.makedirs(current_epoch_save_path, exist_ok=True)
    accelerator.wait_for_everyone() 
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(current_epoch_save_path)
    tokenizer.save_pretrained(current_epoch_save_path)
    print(f"Saved model for Epoch {epoch+1} to {current_epoch_save_path}.")


    # --- Early Stopping Logic & Best Model Saving ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_epoch = epoch + 1 # Update best epoch
        print(f"Validation loss improved! Saving best model from Epoch {best_epoch} to {os.path.join(OUTPUT_DIR, 'best_model')}.")
        # Save model only when validation loss improves, overwriting previous best
        accelerator.wait_for_everyone() 
        unwrapped_model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model")) # Save to a consistent 'best_model' folder
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
    else:
        epochs_without_improvement += 1
        print(f"Validation loss did not improve. Epochs without improvement: {epochs_without_improvement}")
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered! Validation loss did not improve for {EARLY_STOPPING_PATIENCE} epochs. Best model was from Epoch {best_epoch}.")
            break # Exit the training loop
            
    model.train() # Set model back to training mode for next epoch

print("Training complete!")

# --- 7. Load and Test the Fine-tuned Model ---
print("\n--- Testing the Fine-tuned Model ---")

# The device for inference is the one set up by Accelerator during training
inference_device = accelerator.device 
print(f"Inference device: {inference_device}")

# Load the base model for inference (using the same dtype as during training)
base_model_for_inference = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=model_load_dtype,
    device_map="auto", # or inference_device if you prefer
    trust_remote_code=True
)

try:
    # Prioritize loading the 'best_model' if it exists, otherwise fall back to the last epoch's save if training was interrupted mid-way
    final_model_path_for_inference = os.path.join(OUTPUT_DIR, "best_model")
    if not (os.path.exists(final_model_path_for_inference) and os.path.isdir(final_model_path_for_inference) and os.listdir(final_model_path_for_inference)):
        # If best_model doesn't exist (e.g. training interrupted before first improvement, or early stopping patience was 0)
        # Find the latest saved epoch
        latest_epoch_num = -1
        for d in os.listdir(OUTPUT_DIR):
            if d.startswith("epoch_") and os.path.isdir(os.path.join(OUTPUT_DIR, d)):
                try:
                    epoch_num = int(d.split("_")[1])
                    if epoch_num > latest_epoch_num:
                        latest_epoch_num = epoch_num
                except ValueError:
                    continue
        if latest_epoch_num != -1:
            final_model_path_for_inference = os.path.join(OUTPUT_DIR, f"epoch_{latest_epoch_num}")
            print(f"Best model not found. Loading latest completed epoch model from {final_model_path_for_inference}.")
        else:
            raise FileNotFoundError(f"No best model or completed epoch models found in {OUTPUT_DIR}. Falling back to base model.")

    model_for_inference = PeftModel.from_pretrained(base_model_for_inference, final_model_path_for_inference)
    # Merge adapters for faster inference
    model_for_inference = model_for_inference.merge_and_unload()
    model_for_inference.to(inference_device)
    print("Fine-tuned model (adapters) loaded and merged for inference.")

except Exception as e:
    print(f"Error loading fine-tuned model for inference: {e}")
    print("Attempting to load base model only. Inference might be less specialized.")
    model_for_inference = base_model_for_inference.to(inference_device)

model_for_inference.eval()

def generate_bash_script(prompt_text, model_obj, tokenizer_obj, device_obj):
    """Generates a Bash script from a natural language prompt using the fine-tuned model."""
    formatted_prompt = f"### Instruction:\n{prompt_text}\n\n### Response:\n"
    
    inputs = tokenizer_obj(formatted_prompt, return_tensors="pt", return_attention_mask=False)
    inputs = {k: v.to(device_obj) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_obj.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer_obj.eos_token_id,
            pad_token_id=tokenizer_obj.pad_token_id,
        )

    generated_text = tokenizer_obj.decode(outputs[0], skip_special_tokens=True)
    
    try:
        bash_script_start_marker = "```bash\n"
        bash_script_end_marker = "```"

        start_index = generated_text.find(bash_script_start_marker)
        if start_index != -1:
            start_index += len(bash_script_start_marker)
            end_index = generated_text.find(bash_script_end_marker, start_index)
            if end_index != -1:
                return generated_text[start_index:end_index].strip()
            else:
                return generated_text[start_index:].strip()
        else:
            response_marker = "### Response:"
            response_start_index = generated_text.find(response_marker)
            if response_start_index != -1:
                return generated_text[response_start_index + len(response_marker):].strip()
            else:
                return generated_text.strip()
    except Exception as e:
        print(f"Error extracting script from generated text: {e}")
        return generated_text.strip()


# Test prompts
test_prompts = [
    "Create a new directory called 'my_project_data'",
    "Delete all '.txt' files in the current directory",
    "Display the last 10 lines of 'logfile.txt'",
    "List all processes currently running on the system",
    "Count the number of lines in 'input.csv'",
    "Create an empty file named 'report.log'",
    "Download a file from 'https://example.com/file.zip' using curl and save it as 'downloaded_file.zip'",
    "Search for the word 'error' in 'app.log' and print matching lines",
    "Change the permissions of 'script.sh' to be executable by the owner and readable by others",
    "Backup the 'documents' folder to a gzipped tar archive named 'documents_backup.tar.gz'"
]

for i, prompt in enumerate(test_prompts):
    print(f"\n--- Test Case {i+1} ---")
    print(f"Prompt: {prompt}")
    generated_script = generate_bash_script(prompt, model_for_inference, tokenizer, inference_device)
    print(f"Generated Script:\n{generated_script}")

print("\nFine-tuning and testing script finished.")
