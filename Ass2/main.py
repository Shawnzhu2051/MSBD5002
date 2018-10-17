import csv
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

NOMIAL_COL = [1,3,5,6,7,8,9,13]
NUM_COL = [0,2,4,10,11,12]

def csv_reader(FILENAME):
    data = []
    with open(FILENAME) as f:
        reader = csv.reader(f)
        for row in reader:
            line = []
            for item in row:
                if item != '':
                    line.append(item)
                if item == '?':
                    line.append('')
            data.append(line)
    return data[1:]


def preprocess(dataset):
    dataset = type_transform(dataset)
    dataset = normalization(dataset)


def type_transform(dataset):
    category_list = []
    for _index1 in range(len(dataset[0])):
        col = [temp[_index1] for temp in dataset]
        category = {}
        counter = 1
        for item in col:
            if item not in category:
                category[item] = counter
                counter += 1
        category_list.append(category)
    for _index2, line in enumerate(dataset):
        for _index3, item in enumerate(line):
            if _index3 in NOMIAL_COL:
                dataset[_index2][_index3] = category_list[_index3][item]
            else:
                dataset[_index2][_index3] = int(dataset[_index2][_index3])
    return dataset


def normalization(dataset):
    max_min_list = []
    for _index1 in range(len(dataset[0])):
        col = [temp[_index1] for temp in dataset]
        max = col[0]
        min = col[0]
        for item in col:
            if item > max:
                max = item
            if item < min:
                min = item
        max_min_list.append((max,min))
    for _index2, line in enumerate(dataset):
        for _index3, item in enumerate(line):
            dataset[_index2][_index3] = (dataset[_index2][_index3]-max_min_list[_index3][1])/(max_min_list[_index3][0] - max_min_list[_index3][1])
    return dataset


if __name__ == "__main__":

    training_dataset = csv_reader('data/trainFeatures.csv')
    training_labelset = csv_reader('data/trainLabels.csv')

    preprocess(training_dataset)
