import re

def fix_trailing_commas(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove trailing commas in objects or arrays
    fixed_content = re.sub(r',\s*([\]}])', r'\1', content)

    # Write the corrected file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(fixed_content)

# Path to the faulty file
file_path = r"C:\example_path\filtered_data_combined.json"
fix_trailing_commas(file_path)
