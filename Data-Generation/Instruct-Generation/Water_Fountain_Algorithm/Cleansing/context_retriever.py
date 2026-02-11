import os
import json
import requests
import numpy as np

# Configuration, needs to be adjustet to own needs
output_dir = 'c:/example_directory/diversity'
pre_dir = 'c:/example_directory/diversity'
final_dir = os.path.join(pre_dir, 'final')
qa_training_data_path_template = os.path.join(final_dir, 'qaComm_training_data_part_{}.json')
qa_chunk_data_path_template = os.path.join(final_dir, 'qaComm_chunk_data_part_{}.json')

# Load existing JSON files
def load_existing_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    return []

# Global variables for part handling
qa_training_data = []
qa_chunk_data = []
current_qa_training_size = 0
current_qa_chunk_size = 0
training_part_number = 1
chunk_part_number = 1
max_file_size = 7 * 1024 * 1024 * 1024  # 7 GB

# Global ID variable
qa_id = 293294

# Function for writing JSON with file size control
def write_json_to_limited_file(data, template_path, part_number):
    file_path = template_path.format(part_number)
    # Write the data in write mode ('w') instead of append mode ('a')
    if data:  # Only write if data exists
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    # Check the file size
    if os.path.getsize(file_path) >= max_file_size:
        return True  # Indicates the file size exceeded the limit
    return False

# Search the output directory for JSON files
for filename in os.listdir(output_dir):
    if filename.endswith('.json') and filename != 'final':
        file_path = os.path.join(output_dir, filename)
        print(f"Process file: {file_path}")

        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            questions = []
            qa_items = []

            print(f"Found gross questions: {len(data)}")
            for item in data:
                question = item.get('question')
                answer = item.get('answer')
                sources = item.get('sources', [])

                # Check whether at least three sources of SearXNG are available
                searxng_sources = [source for source in sources if source.get('source_info', {}).get('source') == 'SearXNG']

                if question and answer and len(searxng_sources) > 2:
                    if answer not in [
                        "Es tut mir leid, aber es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.",
                        "Es tut mir leid, aber der Kontext ist nicht ausreichend.",
                        " Es tut mir leid, aber es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.",
                        " Es tut mir leid, aber der Kontext ist nicht ausreichend."
                    ]:
                        questions.append(question)
                        qa_items.append((question, answer, searxng_sources))

            print(f"Valid questions found: {len(questions)}")
            print(f"Questions with insufficient context, too few sources or server errors: {len(data) - len(questions)}")

            for question, answer, searxng_sources in qa_items:
                qa_training_data.append({
                    'id': qa_id,
                    'question': question,
                    'answer': answer
                })

                # Save sources
                for source in searxng_sources:
                    qa_chunk_data.append({
                        'id': qa_id,
                        'text': source.get('text'),
                        'url': source.get('source_info', {}).get('url')
                    })

                # Increase ID
                qa_id += 1

            # Check the file size after each save
            print(f"Save part {training_part_number} of qa_training_data...")
            if write_json_to_limited_file(qa_training_data, qa_training_data_path_template, training_part_number):
                # If the file size exceeded the limit, increment the part number
                training_part_number += 1
            #qa_training_data = []
            current_qa_training_size = 0

            print(f"Save part {chunk_part_number} of qa_chunk_data...")
            if write_json_to_limited_file(qa_chunk_data, qa_chunk_data_path_template, chunk_part_number):
                # If the file size exceeded the limit, increment the part number
                chunk_part_number += 1
                qa_chunk_data = []
            current_qa_chunk_size = 0

            print(f"File {filename} processed successfully.")

# Save remaining data
if qa_training_data:
    print(f"Save remaining training data in part {training_part_number}...")
    write_json_to_limited_file(qa_training_data, qa_training_data_path_template, training_part_number)

if qa_chunk_data:
    print(f"Save remaining chunk data in part {chunk_part_number}...")
    write_json_to_limited_file(qa_chunk_data, qa_chunk_data_path_template, chunk_part_number)

print("Data processing completed. Results saved.")
