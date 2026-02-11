import json
import os
import re

# Function for loading JSON data from a file
def load_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function for saving the filtered data in a filei
def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# Function for normalising questions (removes punctuation and makes everything small)
def normalize_text(text):
    # Removes all non-alphabetic characters and sets everything to lower case letters
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation
    return text.lower()  # Converts to lower case

# Function for filtering the question-answer pairs
def filter_data(input_files):
    filtered_data = []
    seen_questions = set()  # Set for questions already seen
    duplicate_count = 0  # Counter for duplicates
    
    seed_questions_json = load_json("c:/example_path/seed_questions.json") # Change the path to the seed questions file if wanted
    seed_questions = [item["frage"] for item in seed_questions_json]

    # Filter criterion: Answer must not contain "I'm sorry" (ignore upper/lower case)s
    forbidden_string = "es tut mir leid"  # Lower case for case-insensitive comparison
    
    # Load and filter data from all filess
    for input_path in input_files:
        data = load_json(input_path)
        
        for entry in data:
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            
            # If the answer does not contain "I'm sorry" (regardless of capitalisation)
            if forbidden_string.lower() not in answer.lower() and question not in seed_questions:
                # Normalise the question (remove punctuation and put everything in lower case)
                normalized_question = normalize_text(question)
                
                # Check whether the normalised question is already in the questions seen (exact match)
                if normalized_question not in seen_questions:
                    filtered_data.append(entry)
                    seen_questions.add(normalized_question)
                else:
                    duplicate_count += 1  # Increase counter if duplicate found
    
    return filtered_data, duplicate_count

# Main function
def main():
    # Input files
    input_files = [
        "c:/example_path/qa_training_data_part_1.json",
        "c:/example_path/qaComm_training_data_part_1.json"
    ]
    
    # Retrieve filtered data
    filtered_data, duplicate_count = filter_data(input_files)
    
    # Output file path
    output_path = "c:/example_path/final_qa.json"
    
    # Save filtered data
    save_json(filtered_data, output_path)
    print(f"Filtered data was saved under: {output_path}")
    print(f"Number of deduplicated entries: {duplicate_count}")

# Call up main function
if __name__ == '__main__':
    main()
