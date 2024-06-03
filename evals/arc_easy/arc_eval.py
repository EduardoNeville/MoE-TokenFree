from datasets import load_dataset
import csv
# Load the dataset from Hugging Face
dataset = load_dataset("ibragim-bad/arc_easy")
# Open a CSV file for writing
with open('evals/arc_easy/data/test/arc_easy_test.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Iterate over the dataset rows
    for split_name, split_data in dataset.items():
        print(f"Parsing {split_name}... \n")
        for row in split_data:
            question = row['question'].replace(","," ")
            choices = row['choices']['text']
            answerKey = row['answerKey']
            
            # Write the row to the CSV file
            writer.writerow([question] + choices + [answerKey])
print("CSV file created successfully.")
