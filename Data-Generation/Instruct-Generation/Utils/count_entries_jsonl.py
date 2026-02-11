import json

# Function for counting the entries in the JSONL file
def count_entries(jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        # Count the number of lines (each line corresponds to a JSON object)s
        count = sum(1 for _ in file)
    return count

# Main function
def main():
    # Path to the JSONL file
    jsonl_file = "c:/example_path/convo_qa.jsonl"
    
    # Count entries
    total_entries = count_entries(jsonl_file)
    
    # Output result
    print(f"Die JSONL-Datei enthält {total_entries} Einträge.")

# Call up main functions
if __name__ == "__main__":
    main()
