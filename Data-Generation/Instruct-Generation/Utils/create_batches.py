import os
import json
from math import ceil
import random

# Function for loading JSON files from a directory
def load_json_files(input_dir):
    data = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
                data.extend(json.load(file))
    random.shuffle(data)
    return data

# Function for splitting the data into batches with an exact number
def split_into_batches(data, num_batches):
    """Creates exactly num_batches batches, even if len(data) is not divisible by num_batches."""
    batches = [[] for _ in range(num_batches)]
    
    # Round-robin procedure: Distributes the elements to the batches one after the other
    for index, item in enumerate(data):
        batch_index = index % num_batches  # Distribute evenly across all batches
        batches[batch_index].append(item)
    
    return batches

# Function for saving the batches in JSON files
def save_batches(batches, output_dir):
    for i, batch in enumerate(batches):
        with open(os.path.join(output_dir, f'example_name_structure{i + 1}.json'), 'w', encoding='utf-8') as file: # Specify Name Structure here
            json.dump(batch, file, ensure_ascii=False, indent=4)

# Hauptfunktion
def main():
    input_dir = 'c:/example_directory/input'
    output_dir = 'c:/example_directory/output'
    num_batches = 59  # Specify number of batches

    data = load_json_files(input_dir)
    batches = split_into_batches(data, num_batches)
    save_batches(batches, output_dir)

if __name__ == '__main__':
    main()
