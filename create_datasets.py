# TODO: Create test pickle file
# TODO: Create train pickle file
# TODO: Create val pickle file

import pickle
import torch
import PIL
import os
import numpy as np
from matplotlib.pyplot import imshow, show
import random
from torchvision import transforms
from PIL.ImageEnhance import Contrast, Brightness


# data format {image: numpy ndarray, label: np.ndarray)


class DataSet():
    def __init__(self, data=None):
        if data is None:
            self.data = {"train": {"images":[],"labels":[]}, "test": {"images": [], "labels": []}, "val": {"images": [],"labels": []}}
        else:
            self.data = data

    def sample_batch(self, size: int, image_set: str):
        labels = []
        images = []
        for item in random.sample(self.data[image_set], size):
            labels.append(item["label"])
            images.append(item["image"])
        return images, labels

    def shuffled_batches(self, image_set: str, batchsize: int = 1):
        images = np.array(self.data[image_set]["images"])
        labels = np.array(self.data[image_set]["labels"])
        seed = random.random()
        random.seed(seed)
        random.shuffle(images)
        random.seed(seed)
        random.shuffle(labels)
        if batchsize == 0:
            return images, labels
        # split into batches
        images = np.array_split(images,len(labels)//batchsize)[:-1]
        labels = np.array_split(labels, len(labels) // batchsize)[:-1]
        batches = []
        for i in range(len(labels)):
            batches.append((torch.tensor(images[i]), torch.tensor(labels[i]).long()))
        return batches

    def add(self, item, image_set, type):
        self.data[image_set][type].append(item)

    def save(self, save_location):
        with open(save_location, "wb") as save_f:
            pickle.dump(self.data, save_f)

    def load(self, save_location):
        with open(save_location, "rb") as save_f:
            self.data = pickle.load(save_f)


class CustomTransforms():
    def random_rotate_image(self, image):
        return image.rotate(random.randint(-30, 30))

    def random_mirror_image(self, image):
        if random.randint(0, 1):
            return image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        else:
            return image

    def change_contrast(self, images):
        res = []
        for image in images:
            enhancer = Contrast(image)
            res.append(image)
            res.append(enhancer.enhance(1 + random.random() / 2))
            res.append(enhancer.enhance(1 + random.random() / 2))
        return res

    def change_brightness(self, images):
        res = []
        for image in images:
            enhancer = Brightness(image)
            res.append(image)
            res.append(enhancer.enhance(1 + random.random() / 2))
            res.append(enhancer.enhance(1 + random.random() / 2))
        return res

    def rescale_image(self, image, factors: tuple):
        return image.resize(factors)

    def to_numpy(self, image):
        return np.array(image)


def create_data(colour_transforms):
    label_translator = {"NORMAL": 0, "PNEUMONIA": 1}
    transformer = CustomTransforms()
    dataset = DataSet()
    for image_set in os.listdir(os.curdir + "/chest_xray"):
        print(image_set)
        for diagnose in os.listdir(os.curdir + "/chest_xray/" + image_set):
            for image_name in os.listdir(os.curdir + "/chest_xray/" + image_set + "/" + diagnose):
                image = PIL.Image.open(os.curdir + "/chest_xray/" + image_set + "/" + diagnose + "/" + image_name).convert("L")
                image = transformer.rescale_image(image, (128, 128))
                if image_set == "train":
                    # perform augmentation

                    # image = transformer.random_rotate_image(image)
                    # image = transformer.random_mirror_image(image)
                    if colour_transforms:
                        images = transformer.change_brightness([image])
                        images = transformer.change_contrast(images)
                    else:
                        images = [image]
                    for img in images:
                        image_tensor = transformer.to_numpy(img).reshape(1, 128, 128)
                        dataset.add(image_tensor, "train", "images")
                        dataset.add(label_translator[diagnose], "train", "labels")
                else:
                    image_tensor = transformer.to_numpy(image).reshape(1, 128, 128)
                    dataset.add(image_tensor, "test", "images")
                    dataset.add(label_translator[diagnose], "test", "labels")

    # Split test and val into equal sized datasets
    imgs, labs = dataset.data["test"]["images"], dataset.data["test"]["labels"]
    random.shuffle(imgs)
    random.shuffle(labs)
    dataset.data["test"]["images"], dataset.data["test"]["labels"], dataset.data["val"]["images"], dataset.data["val"]["labels"] = imgs[:len(imgs) // 2],labs[:len(labs) // 2], imgs[len(imgs) // 2:], labs[:len(labs) // 2:]
    # DEBUG INFO

    dataset.save("pneumonia_data_pickled")


if __name__ == "__main__":
    # this is just a test whether loading works
    create_data(True)
    dataset = DataSet()
    dataset.load("pneumonia_data_pickled")
    a = dataset.shuffled_batches(image_set="test", batchsize=8)
    # for data in a:
    #     images, labels = data
    #     print(images.shape)
