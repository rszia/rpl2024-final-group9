import json

def extract_first_n_objects(input_path, output_path, n=100):
    with open(input_path, 'r') as infile:
        data = json.load(infile)  # data should be a list of objects
    
    # Extract the first n items
    subset = data[:n]
    
    # Write the subset to the output file
    with open(output_path, 'w') as outfile:
        json.dump(subset, outfile, indent=2)

# Example usage
input_file = "R2R_train.json"
output_file = "R2R_train_first_100.json"
extract_first_n_objects(input_file, output_file, 100)
print(f"The first 100 objects have been written to {output_file}.")

input_file = "R2R_test.json"
output_file = "R2R_test_first_100.json"
extract_first_n_objects(input_file, output_file, 100)
print(f"The first 100 objects have been written to {output_file}.")

input_file = "R2R_val_seen.json"
output_file = "R2R_val_seen_first_100.json"
extract_first_n_objects(input_file, output_file, 100)
print(f"The first 100 objects have been written to {output_file}.")

input_file = "R2R_val_unseen.json"
output_file = "R2R_val_unseen_first_100.json"
extract_first_n_objects(input_file, output_file, 100)
print(f"The first 100 objects have been written to {output_file}.")
