import os
import json

# Define directories and individual files
input_paths = [
    'c:/example_file/qa_based_on_chunk.jsonl',
    'c:/example_directory/diversity'
]
# Counter for entries
total_entries = 0

def count_jsonl_entries(file_path):
    global total_entries
    print(f"Process JSONL file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        file_entries = sum(1 for _ in file)
        total_entries += file_entries
        print(f"Number of entries in {file_path}: {file_entries}")

def count_json_entries(file_path):
    global total_entries
    print(f"Process JSON file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        file_entries = len(data)
        total_entries += file_entries
        print(f"Number of entries in {file_path}: {file_entries}")

# Processing of the specified paths
for path in input_paths:
    if os.path.isdir(path):  # If it is a directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if filename.endswith('.json'):
                count_json_entries(file_path)
            elif filename.endswith('.jsonl'):
                count_jsonl_entries(file_path)
    elif os.path.isfile(path):  # If it is a single file
        if path.endswith('.jsonl'):
            count_jsonl_entries(path)
        elif path.endswith('.json'):
            count_json_entries(path)
    else:
        print(f"Path not found or invalid: {path}")

# Output total result
print("\n----------------------")
print("Total result of all files and folders:")
print(f"Total number of entries: {total_entries}")