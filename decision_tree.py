from sklearn.tree import DecisionTreeClassifier
from create_datasets import DataSet
import numpy as np
import time

tree_config = {"max_depth":list(range(3,35,5)),
               "min_samples_leaf":list(range(1,50,5)),
               }


whole_dataset = DataSet()
whole_dataset.load(r"C:\Users\Matej\PycharmProjects\pneumonia_detection\pneumonia_data_pickled")
train_data_images, train_data_labels = whole_dataset.shuffled_batches("train", batchsize=0)
test_data_images, test_data_labels = whole_dataset.shuffled_batches("test", batchsize=0)
val_data_images, val_data_labels = whole_dataset.shuffled_batches("val", batchsize=0)

train_X = [img.flatten() for img in train_data_images]
train_y = train_data_labels
print(len(train_y))

test_X = [img.flatten() for img in test_data_images]
test_y = test_data_labels

val_X = [img.flatten() for img in val_data_images]
val_y = val_data_labels

all_X = np.concatenate((train_X, test_X, val_X))
all_y = np.concatenate((train_y, test_y, val_y))


'''
max_depht=13, min_samples_leaf=26
max_depth=13, min_samples_leaf=31, score=0.835, total=12.5min
max_depth=13, min_samples_leaf=41, score=0.835, total=13.0min
'''


def comb(poss_list):
    if type(poss_list) == dict:
        poss_list = list(poss_list.values())
    start = [[i] for i in poss_list[0]]
    i = 1
    while i < len(poss_list):
        new = []
        for thing in start:
            for item in poss_list[i]:
                new.append(thing + [item])
        start = new
        i += 1
    return start


def validate_alg(classifier):
    correct = 0
    for sample, label in zip(val_X, val_y):
        prediction = classifier.predict([sample])
        if prediction == label:
            correct += 1
        else:
            pass
            # possible to add samples that the algorithm did not guess
    return correct / len(val_y)


def validate_on_training(classifier):
    correct = 0
    for sample, label in zip(train_X[:1000], train_y[:1000]):
        prediction = classifier.predict([sample])
        if prediction == label:
            correct += 1
        else:
            pass
            # possible to add samples that the algorithm did not guess
    return correct / len(train_y[:1000])

best_guys = []
max_score = 0
best_guy = None
config = {"max_depth": list(range(10, 20)), "min_samples_leaf": list(range(2, 20))}
for setup in comb(config):
    dicc = dict(zip(config.keys(),setup))
    tree = DecisionTreeClassifier(**dicc)
    start_time = time.time()
    tree.fit(train_X,train_y)

    train_time = time.time() - start_time
    score = validate_alg(tree)
    train_data_score = validate_on_training(tree)
    print(f"setup= {dicc}",f"accuracy= {round(score,3)}", f"train_accuracy= {round(train_data_score, 3)}", f"training_time= {round(train_time,3)}s", end=" ")
    if score > max_score:
        max_score = score
        best_guys.append([tree,score])
        best_guy = tree
        print("new max")
    elif score > 0.75:
        best_guys.append([tree, score])
        print("decent")
    else:
        print("subpar")
print(sorted(best_guys, key=lambda x: x[1])[-10:])
