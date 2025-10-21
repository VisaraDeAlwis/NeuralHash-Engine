# convert_lfw_pairs.py
import csv

INPUT_FILE = "D://FYP//Madusha_ArcFace//Arcface-Verification-System//datasets//LFW//pairs.csv"
OUTPUT_FILE = "D://FYP//Madusha_ArcFace//Arcface-Verification-System//datasets//LFW//pairs_clean.csv"

cleaned_rows = []

with open(INPUT_FILE, 'r', newline='') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # skip header
    for row in reader:
        # remove empty strings from trailing comma
        row = [item.strip() for item in row if item.strip()]
        if len(row) == 3:  # same-person pair
            name, idx1, idx2 = row
            cleaned_rows.append([name, idx1, idx2])
        elif len(row) == 4:  # different-person pair
            name1, idx1, name2, idx2 = row
            cleaned_rows.append([name1, idx1, name2, idx2])
        else:
            print(f"Skipping invalid row: {row}")

# Write cleaned CSV
with open(OUTPUT_FILE, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    # Write header
    writer.writerow(['name1', 'idx1', 'name2', 'idx2'])
    for row in cleaned_rows:
        if len(row) == 3:  # same-person pair → duplicate name
            writer.writerow([row[0], row[1], row[0], row[2]])
        else:  # different-person pair
            writer.writerow(row)

print(f"✅ Cleaned CSV saved to {OUTPUT_FILE}")
