import os
from google import genai
from pathlib import Path
from collections import defaultdict

# === CONFIGURATION ===
INPUT_FILE = "prompts.txt"
OUTPUT_DIR = "output_round_2"
FAILED_LOG = "failed_prompts.txt"
GENERATED_LOG = "generated_files.txt"
MODEL = "gemini-2.0-flash"

# === SETUP ===
client = genai.Client(api_key="xxx")  # Replace this with your key
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load prompts ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Track used filenames (without extension)
used_filenames = defaultdict(int)

# === Process prompts ===
with open(FAILED_LOG, "w", encoding="utf-8") as failed_log, \
     open(GENERATED_LOG, "w", encoding="utf-8") as gen_log:

    for idx, prompt in enumerate(prompts, 1):
        print(f"[{idx}/{len(prompts)}] Sending request...")

        full_prompt = (
            prompt
            + "\n\nAdd an appropriate filename on the first line in the format '# filename: name.ext', and then comments explaining the script’s purpose below. Only return the comments and the code — no extra text."
        )

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=full_prompt
            )

            reply = response.text.strip()
            lines = reply.splitlines()

            # Defaults
            base_filename = f"script"
            extension = ".sh"
            content_start = 0

            # Try extracting filename
            for i, line in enumerate(lines[:5]):
                if "# filename:" in line.lower():
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        name = parts[1].strip()
                        if "." in name:
                            base_filename, extension = os.path.splitext(name)
                        else:
                            base_filename = name
                    content_start = i + 1
                    break

            # Skip shebang if present
            if lines and lines[0].strip().startswith("#!"):
                content_start = max(content_start, 1)

            # Ensure uniqueness
            final_base = base_filename
            count = used_filenames[base_filename]
            while True:
                suffix = f"_{count+1}" if count > 0 else ""
                candidate_name = f"{idx}.{final_base}{suffix}{extension}"
                output_path = Path(OUTPUT_DIR) / candidate_name
                if not output_path.exists():
                    break
                count += 1
            used_filenames[base_filename] = count + 1

            # Clean and write to .sh and .txt
            cleaned_reply = "\n".join(lines[content_start:]).lstrip()

            # Write .sh file
            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(cleaned_reply)

            # Log both
            gen_log.write(f"{output_path.name}\n")
            print(f"✅ Saved: {output_path.name}")

        except Exception as e:
            print(f"❌ Error on prompt {idx}: {e}")
            failed_log.write(f"{idx}: {prompt}\n")
