import json

# Function for counting the entries in the JSON file
def count_entries(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return len(data)

# Main function
def main():
    # Path to the JSON file
    json_file = "c:/example_path/final_qa.json"
    
    # Count entries
    total_entries = count_entries(json_file)
    
    # Output result
    print(f"Die JSON-Datei enthält {total_entries} Einträge.")

# Call up main function
if __name__ == "__main__":
    main()
