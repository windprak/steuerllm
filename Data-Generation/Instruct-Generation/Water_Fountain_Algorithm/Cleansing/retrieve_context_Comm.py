import os
import json

# Configuration, needs to be adjusted to own needs
output_dir = 'c:/example_path/final_output'
final_dir = os.path.join(output_dir, 'final')
qa_training_data_path = os.path.join(final_dir, 'qaBuchungssätze_training_data.json')
qa_chunk_data_path = os.path.join(final_dir, 'qaBuchungssätze_chunk_data.json')

# Load the JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Globale Variablen
qa_training_data = []
qa_chunk_data = []
qa_id = 541431

# Function for writing JSON with file size control
def write_json_to_file(data, file_path):
    if data:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

# Load the JSONL file
file_path = 'c:/example_path/buchungssätze_qa_4o.jsonl' # Needs to be adjusted
data = load_jsonl(file_path)

# Processing the data
for item in data:
    question = item.get('question')
    answer = item.get('answer')
    context = item.get('context', [])

    # Check if answer is not in the error categories
    if answer not in [
        "All endpoints failed",
        " All endpoints failed",
        "Es tut mir leid, aber es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.",
        "Es tut mir leid, aber der Kontext ist nicht ausreichend.",
        " Es tut mir leid, aber es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.",
        " Es tut mir leid, aber der Kontext ist nicht ausreichend."
    ]:
        qa_training_data.append({
            'id': qa_id,
            'question': question,
            'answer': answer
        })

        # Save context chunks
        for source in context:
            qa_chunk_data.append({
                'id': qa_id,
                'text': source.get('text'),
                'url': source.get('source_info', {}).get('url')
            })

        # Increase ID
        qa_id += 1

# Save the processed data
write_json_to_file(qa_training_data, qa_training_data_path)
write_json_to_file(qa_chunk_data, qa_chunk_data_path)

print("Data processing completed. Results saved.")
