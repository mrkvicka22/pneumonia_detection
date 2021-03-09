#TODO: Note tp self. I resized images to 128*128
import torch
import torch.nn as nn
import torch.nn.functional as F
from create_datasets import DataSet
import torch.optim as optim
import time
import os
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)


def load_data(TRAIN_BATCHSIZE=1, TEST_BATCHSIZE=1, VAL_BATCHSIZE=1):
    whole_dataset = DataSet()
    whole_dataset.load(r"C:\Users\Matej\PycharmProjects\pneumonia_detection\pneumonia_data_pickled")
    train_data = whole_dataset.shuffled_batches("train", TRAIN_BATCHSIZE)
    test_data = whole_dataset.shuffled_batches("test", TEST_BATCHSIZE)  # leave this for now
    val_data = whole_dataset.shuffled_batches("val", VAL_BATCHSIZE)
    print("data loaded")
    return train_data, test_data, val_data


'''
conv = [(W−K+2P) / S] + 1
256-5+0/1
pool =
'''


class Net(nn.Module):
    def __init__(self, l1, l2, l3, l4, c1, c2, c3):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm2d(num_features=1)
        self.c3 = c3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, c1, 3)
        self.batch_norm2 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3)
        self.batch_norm3 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 5)
        self.batch_norm4 = nn.BatchNorm2d(c3)
        self.fc1 = nn.Linear(c3 * 29 * 29, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.fc4 = nn.Linear(l4, 2)

    def forward(self, x):
        x = self.batch_norm(x.float())
        x = self.pool(F.relu(self.batch_norm2(self.conv1(x.float()))))
        x = self.pool(F.relu((self.batch_norm3(self.conv2(x.float())))))
        x = self.pool(F.relu(self.batch_norm4(self.conv3(x))))
        #print("shape:",x.shape)
        x = x.view(-1, self.c3 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.fc4(x)
        return x


def train_pneu(config, checkpoint_dir=None):
    start_time = time.time()
    net = Net(config["l1"], config["l2"], config["l3"], config["l4"], config["c1"], config["c2"], config["c3"])
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=config["momentum"])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",patience=2, verbose=True,factor=0.5)

    print("started batching up")
    batch_time = time.time()
    trainset, testset, valset = load_data(config["train_batch"], config["test_batch"], config["val_batch"])
    print("batching up took", time.time() - batch_time)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainset, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valset, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()  # in case it is running on a gpu. Becuase numpy does not support gpu training
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        lr_scheduler.step(val_loss)

    print('Finished Training', "time:", time.time() - start_time)


def test_accuracy(net, device="cpu"):
    train_set, test_set, val_set = load_data()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return correct / total


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    # config = {
    #     "l1": tune.choice([(2 ** i) * 10 for i in range(8)]),
    #     "l2": tune.choice([(2 ** i) * 10 for i in range(8)]),
    #     "l3": tune.choice([(2 ** i) * 10 for i in range(8)]),
    #     "l4": tune.choice([(2 ** i) * 10 for i in range(8)]),
    #     "c1": tune.choice([2 ** i for i in range(1, 6)]),
    #     "c2": tune.choice([2 ** i for i in range(1, 6)]),
    #     "c3": tune.choice([2 ** i for i in range(1, 6)]),
    #     "lr": tune.choice([1e-4]),
    #     "train_batch": tune.choice([4, 8, 16, 32]),
    #     "test_batch": tune.choice([2]),
    #     "val_batch": tune.choice([2]),
    #     "epochs": tune.choice([2, 4, 6, 8, 10, 20]),
    #     "momentum": tune.choice([i / 10 for i in range(11)])
    # }
    config = {
        "l1": 256,
        "l2": 512,
        "l3": 512,
        "l4": 256,
        "c1": 4,
        "c2": 16,
        "c3": 32,
        "lr": tune.choice([1e-3]),
        "train_batch": tune.choice([4,8,16,32]),
        "test_batch": 2,
        "val_batch": 2,
        "epochs": 80,
        "momentum": 0.9
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=[item for item in config if "batch" not in item] + ["train_batch"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_pneu),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    config = best_trial.config
    best_trained_model = Net(config["l1"], config["l2"], config["l3"], config["l4"], config["c1"], config["c2"],
                             config["c3"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=6, max_num_epochs=80, gpus_per_trial=1)
