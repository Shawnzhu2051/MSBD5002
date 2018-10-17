import csv
import numpy as np
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot

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
    return data


def preprocess(dataset):
    dataset = type_transform(dataset)
    dataset = normalization(dataset)
    return dataset


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


def attribute_adjust(X_train, X_test, y_train, y_test):
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    init_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                               colsample_bytree=1, gamma=0, learning_rate=0.1,
                               max_delta_step=0, max_depth=3, min_child_weight=1,
                               missing=None, n_estimators=100, n_jobs=1,
                               nthread=None, objective='binary:logistic', random_state=0,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                               seed=None, silent=True, subsample=1)

    cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    other_params = {'learning_rate': 0.05, 'n_estimators': 400, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.6, 'colsample_bytree': 0.9, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

    model = XGBClassifier(**other_params)

    grid_search = GridSearchCV(model, cv_params, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train_np, y_train_np)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


def standard_evaluation(X_train, X_test, y_train, y_test):
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                               colsample_bytree=0.9, gamma=0.1, learning_rate=0.1,
                               max_delta_step=0, max_depth=3, min_child_weight=1,
                               missing=None, n_estimators=450, n_jobs=1,
                               nthread=None, objective='binary:logistic', random_state=0,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                               seed=None, silent=True, subsample=0.6)
    model.fit(X_train_np, y_train_np)

    y_pred = model.predict(X_test)

    acc = 0
    for index in range(len(y_pred)):
        if [y_pred[index]] == y_test[index]:
            acc += 1
    acc = acc / len(y_pred)

    print("Accuracy: %.5f%%" % (acc * 100.0))
    # plot_importance(model)
    # pyplot.show()


def predict(X_train, y_train, processed_test):
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                          colsample_bytree=0.9, gamma=0.1, learning_rate=0.1,
                          max_delta_step=0, max_depth=3, min_child_weight=1,
                          missing=None, n_estimators=450, n_jobs=1,
                          nthread=None, objective='binary:logistic', random_state=0,
                          reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                          seed=None, silent=True, subsample=0.6)
    model.fit(X_train_np, y_train_np)

    pred_test = model.predict(processed_test)
    with open('A2_itsc_20567444_prediction.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for line in pred_test:
            writer.writerow(line)


if __name__ == "__main__":

    training_dataset = csv_reader('data/trainFeatures.csv')[1:]
    training_labelset = csv_reader('data/trainLabels.csv')
    test_dataset = csv_reader('data/testFeatures.csv')[1:]

    processed_data = preprocess(training_dataset)
    processed_test = preprocess(test_dataset)

    X = np.array(processed_data)
    Y = np.array(training_labelset)

    seed = 7
    test_size = 0.09
    X_train, X_test, y_train, y_test = train_test_split(processed_data, training_labelset, test_size=test_size, random_state=seed)

    #standard_evaluation(X_train, X_test, y_train, y_test)
    #attribute_adjust(X_train, X_test, y_train, y_test)
    predict(X_train, y_train, processed_test)