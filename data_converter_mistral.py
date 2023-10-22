import json

# Read data from input JSONL file
input_file_path = '/Users/berkgungor/Desktop/IAS/MasterThesis/Codes/golf_chat_data.jsonl'
output_file_path = '/Users/berkgungor/Desktop/IAS/MasterThesis/Codes/mistral_output.jsonl'

# Read data from input JSONL file
def process_jsonl_line(line):
    data = json.loads(line)
    
    # Extract and process the data as needed
    topic = data.get("topic", "")
    conversation = data.get("input", "")
    
    text_between_topic_and_input = topic
    start_index = conversation.find("[|AI|]")
    end_index = conversation.find("[|Human|]", start_index + 1)
    ai_to_human_text = conversation[start_index:end_index].replace("[|AI|]", "").strip()

    # Construct the desired output format
    output_format = {
        'text': f'<|system|>\n You are a Golf Assistant who helps with user queries assistant who always responds in the style of a professional golf trainer.\n<|user|>\n{text_between_topic_and_input}\n<|assistant|>\n{ai_to_human_text}'
    }
    
    return output_format

# Process input JSONL file and write to the output JSONL file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        processed_data = process_jsonl_line(line)
        output_file.write(json.dumps(processed_data) + '\n')