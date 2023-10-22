import json
import re

def convert_data(data, id):
    human_text = data["text"].split("### Human: ")[1].split("### Assistant:")[0].strip()
    assistant_text = data["text"].split("### Assistant:")[0].strip()
    return {"id": str(id), "data": [human_text, assistant_text]}

# Input and output file paths
input_jsonl_file = '../LanguageData/data/GolfData/train.jsonl'  # Replace with your input JSONL file
output_json_file = 'output.json'  # Replace with your desired output JSON file

# Read data from the JSONL file
with open(input_jsonl_file, 'r') as jsonl_file:
    jsonl_lines = jsonl_file.readlines()

# Convert and store data in a list
converted_data = [convert_data(json.loads(line), id) for id, line in enumerate(jsonl_lines)]

# Write the converted data to a JSON file
with open(output_json_file, 'w') as json_file:
    json.dump(converted_data, json_file, indent=4)

print("Conversion complete. Data written to", output_json_file)