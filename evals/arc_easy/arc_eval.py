from datasets import load_dataset
import csv
# Load the dataset from Hugging Face
dataset = load_dataset("ibragim-bad/arc_easy")
# Open a CSV file for writing
with open('evals/arc_easy/data/test/arc_easy.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answerKey'])
    
    # Iterate over the dataset rows
    for split_name, split_data in dataset.items():
        print(f"Parsing {split_name}... \n")
        for row in split_data:
            question = row['question']
            choices = row['choices']['text']
            answerKey = row['answerKey']
            
            # Write the row to the CSV file
            writer.writerow([question] + choices + [answerKey])
print("CSV file created successfully.")
