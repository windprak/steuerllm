import json
import os


# This script is designed to break one questions/input file into multiple, e.g., for the seed questions to create the inital batches.

# Configurable number of chunks
NUM_CHUNKS = 16 # Adjust the desired number of chunks here

# Read file
input_file = 'c:/example_path/seed_questions.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, dict) and 'questions' in data:
    questions = data['questions']
elif isinstance(data, list):
    questions = data
else:
    raise ValueError("The JSON data has an unexpected structure.")

# Add a new element 'type' with the value '' to each question
for question in questions:
    question['typ'] = ""  # Add the element 'type'

# Calculation of the chunk size
total_questions = len(questions)
chunk_size = total_questions // NUM_CHUNKS

# Chunking the questions
chunks = [questions[i * chunk_size:(i + 1) * chunk_size] for i in range(NUM_CHUNKS)]

# Ensure that all questions are in the chunks
if total_questions % NUM_CHUNKS != 0:
    chunks[-1].extend(questions[NUM_CHUNKS * chunk_size:])  # Remaining questions in the last chunks

# Create the new JSON files
output_dir = 'c:/example_path/seed_batches'
for i, chunk in enumerate(chunks, start=1):
    output_file = os.path.join(output_dir, f'example_name_structure{i}.json') # Change name structure
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, indent=4, ensure_ascii=False)

print(f'{NUM_CHUNKS} files have been created.')
