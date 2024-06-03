import csv
# Function to convert number to corresponding letter
def number_to_letter(number):
    return chr(ord('A') + int(number) - 1)
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
