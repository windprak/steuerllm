import os
import json

# Define directories and individual files
input_paths = [
    "c:/example_path/qa_training_data_part_1.json",
    "c:/example_path/qaDiversity_training_data_part_1.json"
]

# Counter for questions
total_valid_questions = 0
total_invalid_questions = 0

def process_jsonl_file(file_path):
    global total_valid_questions, total_invalid_questions
    print(f"Process JSONL file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        valid_questions = 0
        invalid_questions = 0

        for line in file:
            item = json.loads(line.strip())
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
                    valid_questions += 1
            else:
                invalid_questions += 1

        # Output results for the file
        print(f"Valid questions found: {valid_questions}")
        print(f"Questions with insufficient context, too few sources or server errors: {invalid_questions}")
        
        # Update total counter
        total_valid_questions += valid_questions
        total_invalid_questions += invalid_questions

def process_file(file_path):
    global total_valid_questions, total_invalid_questions
    print(f"Process file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        valid_questions = 0
        invalid_questions = 0

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
                    valid_questions += 1
            else:
                invalid_questions += 1


        total_valid_questions += valid_questions
        total_invalid_questions += invalid_questions

        
        # Update total counter


# Processing of the specified paths
for path in input_paths:
    if os.path.isdir(path):  # If it is a directory
        for filename in os.listdir(path):
            if filename.endswith('.json') and filename != 'final':
                process_file(os.path.join(path, filename))
    elif os.path.isfile(path):  # If it is a single file
        if path.endswith('.jsonl'):
            process_jsonl_file(path)
    else:
        print(f"Path not found or invalid: {path}")

# Output total result
print("\n----------------------")
print("Total result of all files and folders:")
print(f"Overall valid questions: {total_valid_questions}")
print(f"Total invalid questions: {total_invalid_questions}")
