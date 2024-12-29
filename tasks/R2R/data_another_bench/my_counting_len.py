## This script is used for counting how many instructions a json file has

import json
import os

# Function to process all JSON files in a folder
def process_json_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    # Traverse all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):  # Only process .json files
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            
            try:
                # Open and load the JSON file
                with open(file_path, 'r') as file:
                    data = json.load(file)
                
                # Ensure the loaded data is a list
                if isinstance(data, list):
                    list_length = len(data)
                    print(f"  - The list contains {list_length} items.")
                else:
                    print(f"  - Error: The file does not contain a list.")
            except json.JSONDecodeError:
                print(f"  - Error: Failed to decode JSON. Check the file format.")
            except Exception as e:
                print(f"  - Error: {e}")

# Example usage
if __name__ == "__main__":
    folder_path = "/home/huang/Desktop/rpl2024-final-group9/Matterport3DSimulator/tasks/R2R/data_another_bench/4o-45-ins"  # Replace with your folder path
    process_json_files_in_folder(folder_path)
