import csv

def csv_reader():
    data = []
    with open('groceries.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            line = []
            for item in row:
                if item != '':
                    line.append(item)
            data.append(line)
    return data

if __name__ == "__main__":

    data = csv_reader()