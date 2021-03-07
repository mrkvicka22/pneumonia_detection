from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from create_datasets import DataSet
import numpy as np
whole_dataset = DataSet()
whole_dataset.load(r"C:\Users\Matej\PycharmProjects\pneumonia_detection\pneumonia_data_pickled")
train_data_images, train_data_labels = whole_dataset.shuffled_batches("train", batchsize=0)
test_data_images, test_data_labels = whole_dataset.shuffled_batches("test", batchsize=0)
val_data_images, val_data_labels  = whole_dataset.shuffled_batches("val", batchsize=0)


train_X = [img.flatten() for img in train_data_images]
train_y = train_data_labels

test_X = [img.flatten() for img in test_data_images]
test_y = test_data_labels

val_X = [img.flatten() for img in val_data_images]
val_y = val_data_labels

all_X = np.concatenate((train_X,test_X,val_X))
all_y = np.concatenate((train_y,test_y,val_y))

# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
#
#
#
clf = RandomForestClassifier(n_estimators=100, random_state=0,n_jobs=-1, min_samples_leaf=5,class_weight="balanced")
#clf.fit(val_X, val_y)
scores = cross_val_score(clf, all_X, all_y,n_jobs=-1, cv=10, scoring='precision')
print(scores)
