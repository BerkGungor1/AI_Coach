from datasets import load_dataset
import pandas as pd
import numpy as np

#dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
#dataset = dataset.filter(lambda x:x['context'] == '')

import jsonlines

input_file = "/Users/berkgungor/Desktop/IAS/MasterThesis/Codes/data_converter_mistral_dolly.py"  # Replace with your input file
output_file = "filtered_output.jsonl"  # Replace with your desired output file

with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
    for line in reader:
        if line.get("context"):
            if "category" in line:
                del line["category"]
            writer.write(line)