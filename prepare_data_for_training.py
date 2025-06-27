import os
import re
import json
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split

def extract_bash_data(script_file_path):
    """
    Extracts the leading comment block as a prompt and the rest of the script
    as the completion from a Bash script file.
    Robustly handles shebang lines, blank lines, and common usage patterns.
    """
    with open(script_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    prompt_lines_raw = []
    script_lines = []
    
    # Flags to manage parsing state
    in_prompt_section = True
    found_first_non_empty_line = False
    
    for line_num, line in enumerate(lines):
        stripped_line = line.strip()

        if in_prompt_section:
            if not found_first_non_empty_line and stripped_line == "":
                # Skip initial blank lines before any content
                continue
            
            found_first_non_empty_line = True

            if line.startswith("#!"):
                # Shebang is the start of the script, not the prompt.
                # If we encounter shebang, the prompt section is over.
                in_prompt_section = False
                script_lines.append(line)
            elif stripped_line.startswith("#"):
                # This is a comment line, potential part of the prompt
                prompt_lines_raw.append(line)
            elif stripped_line == "":
                # A blank line after some comments or non-comment lines signals the end of the prompt block
                # if comments were found. If no comments yet, it's just a blank line before code.
                if prompt_lines_raw: # If we already collected some prompt lines, this blank line ends it
                    in_prompt_section = False
                else: # If no prompt lines yet, this blank line is not useful, just skip
                    continue
            else:
                # This is the first non-comment, non-empty line (not a shebang).
                # It means the prompt section has ended, and this line is part of the script.
                in_prompt_section = False
                script_lines.append(line)
        else:
            # All lines after the determined prompt section are part of the script
            script_lines.append(line)
    
    # Process extracted prompt lines
    prompt_cleaned_lines = [line.lstrip('# ').strip() for line in prompt_lines_raw if line.strip()]
    prompt = " ".join(prompt_cleaned_lines).strip()

    # Process script lines. Remove any leading/trailing empty lines that might have been carried over.
    completion = "".join(script_lines).strip()

    # Final sanity check: if the "prompt" is empty (meaning no valid leading comments were found for the prompt)
    # or if the "completion" is empty (meaning the file had only comments or was empty), then skip this example.
    if not prompt or not completion:
        print(f"Warning: Skipping '{os.path.basename(script_file_path)}' due to empty prompt or completion after extraction.")
        return None

    return {"prompt": prompt, "completion": completion}

def process_all_bash_scripts_for_dataset(input_directory, output_jsonl_file, test_size=0.1, random_state=42):
    """
    Processes all .sh files in the input_directory to create the dataset
    and splits it into training and validation sets, using the robust extract_bash_data.
    """
    all_raw_data_points = []
    skipped_count = 0

    print(f"Scanning directory: {input_directory} for Bash scripts...")
    for filename in os.listdir(input_directory):
        if filename.endswith(".sh"): # Assuming your scripts have .sh extension
            script_path = os.path.join(input_directory, filename)
            data = extract_bash_data(script_path)
            if data:
                all_raw_data_points.append(data)
            else:
                skipped_count += 1

    if not all_raw_data_points:
        print(f"Error: No valid Bash scripts with comments found in '{input_directory}' to process.")
        return

    # Split data into training and validation sets
    train_data, val_data = train_test_split(all_raw_data_points, test_size=test_size, random_state=random_state)

    # Combine training and validation data into the final JSONL format
    formatted_training_data = []
    for item in train_data:
        formatted_text = (
            f"### Instruction:\n{item['prompt']}\n\n"
            f"### Response:\n```bash\n{item['completion']}\n```"
        )
        formatted_training_data.append({"text": formatted_text})

    formatted_validation_data = []
    for item in val_data:
        formatted_text = (
            f"### Instruction:\n{item['prompt']}\n\n"
            f"### Response:\n```bash\n{item['completion']}\n```"
        )
        formatted_validation_data.append({"text": formatted_text})

    # Create DatasetDict from the formatted data
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(formatted_training_data),
        'validation': Dataset.from_list(formatted_validation_data)
    })

    # Saving as JSONL. These files are for reference or direct loading with load_dataset('json', ...)
    dataset_dict['train'].to_json(output_jsonl_file.replace(".jsonl", "_train.jsonl"), orient="records", lines=True, force_ascii=False)
    dataset_dict['validation'].to_json(output_jsonl_file.replace(".jsonl", "_val.jsonl"), orient="records", lines=True, force_ascii=False)

    print(f"Extracted {len(all_raw_data_points)} total scripts.")
    print(f"Saved {len(formatted_training_data)} training examples to {output_jsonl_file.replace('.jsonl', '_train.jsonl')}")
    print(f"Saved {len(formatted_validation_data)} validation examples to {output_jsonl_file.replace('.jsonl', '_val.jsonl')}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} scripts due to extraction issues (e.g., no leading comments or empty script content).")

    return dataset_dict # Return the DatasetDict directly for use in the main script

# --- How to use this script ---
if __name__ == "__main__":
    # IMPORTANT: Adjust this path to where your renamed Bash scripts are located
    # Example: input_scripts_dir = "/Users/your_username/Documents/ai/bash_scripts"
    input_scripts_dir = os.path.expanduser("~/Documents/ai/bash_scripts") # This handles '~' for you
    output_dataset_file = "bash_scripts_dataset.jsonl" # This will generate bash_scripts_dataset_train.jsonl and _val.jsonl

    process_all_bash_scripts_for_dataset(input_scripts_dir, output_dataset_file)

    print("\nDataset preparation complete. You can now proceed with the fine-tuning step.")
    print("The generated files are: 'bash_scripts_dataset_train.jsonl' and 'bash_scripts_dataset_val.jsonl'")
    print("These can be loaded directly by Hugging Face `load_dataset` in the next step.")
