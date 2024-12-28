import json
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

developer_prompt_minimal = (
    "Rewrite the instructions in a simpler, more direct style. "
    "Remove descriptive details and keep only essential directions. "
    "Return a single sentence or very short paragraph *without any Markdown formatting* (no bold, no lists). "
    "Do not include quotes or headings like 'Simplified Instructions'. "
    "Just return the transformed instruction."
)

developer_prompt_goal = (
    "Rewrite the instructions to focus on the final goal only. "
    "Remove all intermediate steps. "
    "Return a single short imperative sentence *without any Markdown formatting*. "
    "Do not include quotes or headings. "
    "Just return the transformed instruction."
)

def gpt_transform(instruction, developer_prompt):
    """
    Use o1-model (e.g., o1-mini-2024-09-12) to transform the given instruction
    according to the developer_prompt. 
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="o1-mini-2024-09-12",
        messages=[
            {
                "role": "user", 
                "content": (
                    f"{developer_prompt}\n\n"
                    f"Original instruction:\n"
                    f"\"{instruction}\"\n\n"
                    f"Transform it accordingly:"
                )
            }
        ],
    )
    return response.choices[0].message.content.strip()

def process_r2r_data(input_path, output_path, max_entries=None):
    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_count = 0
    for entry in data:
        if max_entries is not None and processed_count >= max_entries:
            break

        original_instructions = entry.get('instructions', [])
        minimal_instructions = []
        goal_only_instructions = []

        for instr in original_instructions:
            # Transform to minimal instructions
            minimal_instr = gpt_transform(instr, developer_prompt_minimal)
            minimal_instructions.append(minimal_instr)

            # Transform to goal-only instructions
            goal_instr = gpt_transform(instr, developer_prompt_goal)
            goal_only_instructions.append(goal_instr)

        entry["instructions_minimal"] = minimal_instructions
        entry["instructions_goal_only"] = goal_only_instructions

        processed_count += 1

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage: Adjust 'max_entries' if you only want to process a subset.
if __name__ == "__main__":
    input_file = "R2R_test.json"
    output_file = "R2R_test_processed.json"
    process_r2r_data(input_file, output_file, max_entries=100)
    print(f"Processing complete. Check {output_file} for results.")

    input_file = "R2R_train.json"  # Your original R2R data file
    output_file = "R2R_train_processed.json"
    process_r2r_data(input_file, output_file, max_entries=100)

    print(f"Processing complete. Check {output_file} for results.")

    input_file = "R2R_val_seen.json"  # Your original R2R data file
    output_file = "R2R_val_seen_processed.json"
    process_r2r_data(input_file, output_file, max_entries=100)

    print(f"Processing complete. Check {output_file} for results.")

    input_file = "R2R_val_unseen.json"  # Your original R2R data file
    output_file = "R2R_val_unseen_processed.json"
    process_r2r_data(input_file, output_file, max_entries=100)

    print(f"Processing complete. Check {output_file} for results.")
