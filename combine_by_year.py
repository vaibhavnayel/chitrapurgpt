import os
import re
from collections import defaultdict
from datetime import datetime

# Directory paths
input_dir = 'documents/txts'
output_dir = 'temp'

# Dictionary to store files by year
files_by_year = defaultdict(list)

# Read all files and organize them by year
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        match = re.match(r'Sunbeam-(\d{4})-(\w+)\.txt', filename)
        if match:
            year = match.group(1)
            month = match.group(2)
            try:
                # Parse the date to help with sorting
                date = datetime.strptime(f"{year} {month}", "%Y %B")
                files_by_year[year].append((date, os.path.join(input_dir, filename)))
            except ValueError:
                print(f"Skipping file with invalid date format: {filename}")

# Process each year
for year, file_list in files_by_year.items():
    # Sort files by date
    file_list.sort(key=lambda x: x[0])
    
    # Create output file
    output_file = os.path.join(output_dir, f'Sunbeam-{year}.txt')
    
    print(f"Creating {output_file}")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write header
        outfile.write(f"Combined Sunbeam entries for {year}\n")
        outfile.write("=" * 50 + "\n\n")
        
        # Process each file
        for _, filepath in file_list:
            # Write month separator
            filename = os.path.basename(filepath)
            month = filename.split('-')[2].replace('.txt', '')
            outfile.write(f"\n{month}\n")
            outfile.write("-" * len(month) + "\n\n")
            
            # Copy content
            with open(filepath, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
                if not content.endswith('\n'):
                    outfile.write('\n')
                outfile.write('\n')

print("Finished combining files by year") 