import json

# Function to load the input file, process it and create the output file in JSONL format
def convert_to_jsonl(input_file, output_file):
    # Open and load the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as f:
        # Iterate through the data and write each conversation as a separate line
        for entry in data:
            user_question = entry.get("question", "")
            assistant_answer = entry.get("answer", "")
            
            # Jede Frage-Antwort-Kombination als separate Konversation
            conversation = {
                "conversations": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": assistant_answer}
                ]
            }
            
            # Write the JSON object as a line in the JSONL file
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

# Example call of the function
input_file = r'C:\example_path\final_qa.json'
output_file = r'C:\example_path\convo_qa.jsonl'

convert_to_jsonl(input_file, output_file)

print(f"The conversion to JSONL has been completed. The output file is saved in: {output_file}")
