import csv
from datasets import load_dataset
# Load the dataset from Hugging Face
dataset = load_dataset("ibragim-bad/arc_easy")
# Function to convert number to corresponding letter
def number_to_letter(number):
    return chr(ord('A') + int(number) - 1)
# Open a CSV file for writing
with open('evals/arc_easy/data/test/arc_easy_test.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='|')
    
    # Write the header
    writer.writerow(['question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answerKey'])
    
    # Iterate over the dataset rows
    for split_name, split_data in dataset.items():
        print(f"Parsing {split_name}... \n")
        for row in split_data:
            question = row['question'].replace(",", " ")
            choices = row['choices']['text']
            answerKey = row['answerKey']
            
            # If there are more than 4 choices, remove one incorrect choice
            if len(choices) > 4:
                correct_index = ord(answerKey) - ord('A')
                for i in range(len(choices)):
                    if i != correct_index:
                        choices.pop(i)
                        break            
            # Write the row to the CSV file
            writer.writerow([question] + choices + [answerKey])
print("CSV file created successfully.")
# Read the original CSV file
input_file_path = 'evals/arc_easy/data/test/arc_easy_test.csv'
output_file_path = 'evals/arc_easy/data/test/arc_easy_cleaned.csv'
with open(input_file_path, mode='r') as infile, open(output_file_path, mode='w', newline='') as outfile:
    reader = csv.reader(infile, delimiter=',')
    writer = csv.writer(outfile, delimiter=',')
    
    # Iterate over the rows in the CSV file
    for row in reader:
        # Check if the last element is a number
        if row[-1].isdigit():
            # Convert the number to the corresponding letter
            row[-1] = number_to_letter(row[-1])
        
        # Write the cleaned row to the new CSV file
        writer.writerow(row)
print("CSV file cleaned and saved successfully.")
