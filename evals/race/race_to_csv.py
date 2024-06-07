from json import load
from re import split
from datasets import load_dataset
import ast
import os
import csv

data = load_dataset("EleutherAI/race", split="test")

print(f"this is the data: {data}")

csvData = []

# Characters to replace
chars_to_replace = ",'\""
# Create translation table
translation_table = str.maketrans(chars_to_replace, ' ' * len(chars_to_replace))
# Replace characters
counter = 0
for section in data:
    print(f"Progress: {(counter/1045)*100} %")
    counter += 1
    # Convert the string representation of the list of dictionaries to an actual list of dictionaries
    article = section['article'].translate(translation_table)
    problems: list[dict] = ast.literal_eval(section['problems'])
    for problem in problems:
        question = problem['question'].translate(translation_table)
        answerKey = problem['answer'].translate(translation_table)
        ansChoices = [choice.translate(translation_table) for choice in problem['options']]
        questionCSV = f"{article}{question}"
        csvRow = [
            questionCSV,
            ansChoices[0],
            ansChoices[1],
            ansChoices[2],
            ansChoices[3],
            answerKey
        ]
        #print(f"Csv Row created! \n {csvRow}")
        csvData.append(csvRow)

csvDev = csvData[:50]
csvTest = csvData[51:]

# Ensure the directories exist
os.makedirs('evals/race/test', exist_ok=True)
os.makedirs('evals/race/dev', exist_ok=True)
# Write to race_test.csv
with open('evals/race/test/race_test.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Article and Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer'])
    csvwriter.writerows(csvTest)
print("Test csv created!")
# Write to race_dev.csv
with open('evals/race/dev/race_dev.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Article and Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer'])
    csvwriter.writerows(csvDev)
print("Dev csv created!")

