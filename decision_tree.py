from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from create_datasets import DataSet
import numpy as np


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

test_X = [img.flatten() for img in test_data_images]
test_y = test_data_labels

val_X = [img.flatten() for img in val_data_images]
val_y = val_data_labels

all_X = np.concatenate((train_X, test_X, val_X))
all_y = np.concatenate((train_y, test_y, val_y))

# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
#
#
#

tree = DecisionTreeClassifier(random_state=0, min_samples_leaf=5, class_weight="balanced")
# clf.fit(val_X, val_y)
clf = GridSearchCV(tree, tree_config, verbose=3,n_jobs=10)
clf.fit(all_X, all_y)

print(clf.cv_results_,clf.best_params_)
'''
max_depth=13, min_samples_leaf=31, score=0.835, total=12.5min
max_depth=13, min_samples_leaf=41, score=0.835, total=13.0min
'''



