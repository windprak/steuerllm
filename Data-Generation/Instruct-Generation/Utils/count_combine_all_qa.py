import json
import os
import re

# Function for loading JSON data from a file
def load_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Function for normalising questions (removes punctuation and makes everything small)
def normalize_text(text):
    # Removes all non-alphabetic characters and sets everything to lower case letters
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation
    return text.lower()  # Converts to lower case

# Function for filtering the question-answer pairs
def filter_data(input_files):
    
    seen_questions = set()  # Set for questions already seen
    duplicate_count = 0  # Counter for duplicates
    seed_questions_json = load_json("c:/Users/lauri/Documents/Studium/Masterarbeit/Data/seed_questions_CORRECTED.json")
    seed_questions = [item["frage"] for item in seed_questions_json]
    complete_count = 0 
    # Filter criterion: Answer must not contain "I'm sorry" (ignore upper/lower case)
    forbidden_string = "es tut mir leid"  # Lower case for case-insensitive comparison
    complete_dismissed_count = 0
    # Load and filter data from all files
    for input_path in input_files:
        data = load_json(input_path)
        count_questions = 0

        for entry in data:
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            
            # If the answer does not contain "I'm sorry" (regardless of capitalisation)
            if forbidden_string.lower() not in answer.lower() and question not in seed_questions:
                # Normalise the question (remove punctuation and put everything in lower case)
                normalized_question = normalize_text(question)
                
                # Check whether the normalised question is already in the questions seen (exact match)
                if normalized_question not in seen_questions:
                    count_questions += 1
                    seen_questions.add(normalized_question)
                else:
                    duplicate_count += 1  # Increase counter if duplicate found
            else:
                complete_dismissed_count += 1
        print(f"In {input_path} there are: {count_questions} Valid Objects")
        complete_count += count_questions
    print(f"In all Files there are: {complete_count} Valid Objects")
    print(f"In all Files there are: {complete_dismissed_count} Invalid Objects")  
    return duplicate_count

# Main function
def main():
    # Input files
    input_files = [
        "c:/example_path/qa_training_data_part_1.json",
        "c:/example_path/qaDiversity_training_data_part_1.json"
    ]
    
    # Retrieve filtered datas
    duplicate_count = filter_data(input_files)
    
    # Output file path

    print(f"Anzahl der deduplizierten Einträge: {duplicate_count}")

# Call up main function
if __name__ == '__main__':
    main()
