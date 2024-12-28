import json
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Prompts for GPT transformation
system_prompt_minimal = (
    "You are a helpful assistant that transforms detailed navigation instructions into minimal, primarily directional instructions. "
    "You should remove all unnecessary descriptive details about objects, colors, or landmarks that are not needed for navigation. "
    "Keep directions simple and focus only on actions and basic references (like hallway, room, door)."
)

system_prompt_goal = (
    "You are a helpful assistant that transforms detailed navigation instructions into a very short instruction that focuses only on the final goal. "
    "You should remove all intermediate steps and directions, and simply state the final destination in a short imperative sentence."
)


def gpt_transform(instruction, system_prompt):
    client = OpenAI()
    """Use GPT to transform the given instruction according to the provided system prompt."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original instruction:\n{instruction}\n\nTransform it accordingly:"}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def process_r2r_data(input_path, output_path, max_entries=None):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # If data is not a list, adjust accordingly. The R2R data typically is a list of path entries.
    processed_count = 0
    for entry in data:
        if max_entries is not None and processed_count >= max_entries:
            break

        original_instructions = entry.get('instructions', [])
        minimal_instructions = []
        goal_only_instructions = []

        for instr in original_instructions:
            # Transform to minimal instructions
            minimal_instr = gpt_transform(instr, system_prompt_minimal)
            minimal_instructions.append(minimal_instr)

            # Transform to goal-only instructions
            goal_instr = gpt_transform(instr, system_prompt_goal)
            goal_only_instructions.append(goal_instr)

        # Add the new fields to the entry
        entry['instructions_minimal'] = minimal_instructions
        entry['instructions_goal_only'] = goal_only_instructions

        processed_count += 1

    # Write the processed data back to output
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# Example usage:
# Adjust 'max_entries' if you want to limit how many data entries are processed.
input_file = "R2R_test.json"  # Your original R2R data file
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
