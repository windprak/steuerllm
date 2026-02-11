import json

def validate_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json.load(file)
        print(f"{file_path} is a valid JSON.")
    except json.JSONDecodeError as e:
        print(f"Error in {file_path}: {e}")

def validate_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, start=1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error in line {i} of {file_path}: {e}")
                    return
        print(f"{file_path} is a valid JSONL.")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error has occurred: {e}")


# Pfad zur JSON-Datei
file_path = "c:/example_path/convo_qa.jsonl"
if file_path.split(".")[1] == "json":
    validate_json(file_path)
elif file_path.split(".")[1] == "jsonl":
    validate_jsonl(file_path)
