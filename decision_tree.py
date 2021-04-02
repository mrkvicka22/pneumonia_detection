from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import random_split
from dataset_class import PneumoniaDataset
from torchvision import transforms
import numpy as np
import time
from collections import Counter

tree_config = {"max_depth":list(range(3,35,5)),
               "min_samples_leaf":list(range(1,50,5)),
               }

transformations = transforms.Compose([transforms.Grayscale(), transforms.Resize((128, 128)), transforms.ToTensor(),
                                      transforms.Normalize(mean=0.48814950165, std=0.24329058187339847)])
# Call the dataset

whole_dataset = PneumoniaDataset(path=r"C:\Users\Matej\PycharmProjects\pneumonia_detection\data_indexer", transforms=transformations)

#split it into datasets of 70/15/15
lengths = [round(len(whole_dataset)*0.7), round(len(whole_dataset) * 0.15), round(len(whole_dataset) * 0.15)- (round(
    len(whole_dataset) * 0.7) + round(len(whole_dataset) * 0.15) + round(len(whole_dataset) * 0.15) - len(whole_dataset))]
train_dataset,test_dataset,val_dataset = random_split(whole_dataset, lengths)


val_X = [x.flatten().numpy() for x, y in val_dataset]
val_y = [[y] for x, y in val_dataset]

test_X = [x.flatten().numpy() for x, y in test_dataset]
test_y = [[y] for x, y in test_dataset]

train_X = [x.flatten().numpy() for x, y in train_dataset]
train_y = [[y] for x, y in train_dataset]


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
config = {"max_depth": list(range(10, 50,3)), "min_samples_leaf": list(range(5, 155,5)),"class_weight":["balanced"]}
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
