import csv
import json
import random
from tqdm import tqdm
from pathlib import Path
def number_to_letter(number):
    return chr(ord('A') + int(number) - 1)
path = Path('evals/coqa/coqa-dev-v1.0.json')

# Load JSON data
with open(path) as f:
    data = json.load(f)

#print(f"Data loaded: {data['data'][0]['story']}")

# Extract story, questions, and answers
# Prepare CSV data
print(f"Data is ready begin CSV creation...")
csv_data = []
print(f"length of data is {len(data['data'])}")
cumulative = 0
for dt in data['data']:
    cumulative += len(dt)
    story = dt['story'].replace(","," ")
    questions = dt['questions']
    answers = dt['answers']
    # Create a list of all answers
    all_answers = [answer['input_text'] for answer in answers]
    #print(f"all answers: {all_answers}")
    for question in questions:
        question_text: str = question['input_text'].replace(",", " ")
        correct_answer: str = next(answer for answer in answers if answer['turn_id'] == question['turn_id'])['input_text'].replace(",", " ")

        # Select 3 random answers from other questions
        other_answers = [ans for ans in all_answers if ans != correct_answer]
        if len(other_answers) < 3:
            print(f"Not enough other answers to sample from for question: {question_text}")
            continue
        no_correct = random.sample(other_answers, 3)

        other_answers = [ans.replace(",", " ") for ans in no_correct]
        
        # Comeebine correct answer with other answers and shuffle
        
        other_answers.append(correct_answer)
        answer_choices = other_answers
        random.shuffle(answer_choices)
        
        # Determine the correct answer's position
        correct_answer_position = answer_choices.index(correct_answer)
        correct_answer_label = chr(65 + correct_answer_position)  # 'A', 'B', 'C', or 'D'
        question_csv = f"{story}{question_text}"
        
        # Construct the CSV row
        csv_row = [
            question_csv,
            answer_choices[0],
            answer_choices[1],
            answer_choices[2],
            answer_choices[3],
            correct_answer_label
        ]
        #print(f"CSV row: \n {csv_row}")
        
        csv_data.append(csv_row)

# Write to CSV file
with open('evals/coqa/data/dev/coqa_dev.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Story and Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer'])
    csvwriter.writerows(csv_data)

print("CSV file has been created.")
