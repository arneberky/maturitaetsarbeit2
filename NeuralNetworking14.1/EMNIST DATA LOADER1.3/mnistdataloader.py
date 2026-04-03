import os
import numpy as np
from PIL import Image
import PIL.ImageOps
import random
import struct
from pathlib import Path

#1.3

def loadidximages(filepath):
    with open(filepath, 'rb') as openedfile:
        magic, numimages, rows, cols = struct.unpack('>IIII', openedfile.read(16))
        return np.frombuffer(openedfile.read(), dtype=np.uint8).reshape(numimages, rows, cols)

def loadidxlabels(filepath):
    with open(filepath, 'rb') as openedfile:
        magic, numitems = struct.unpack('>II', openedfile.read(8))
        return np.frombuffer(openedfile.read(), dtype=np.uint8)

def loademnistbyclass():
    filepaths = {"train_images": "emnist-byclass-train-images-idx3-ubyte", "train_labels": "emnist-byclass-train-labels-idx1-ubyte", "test_images": "emnist-byclass-test-images-idx3-ubyte", "test_labels": "emnist-byclass-test-labels-idx1-ubyte"}
    
    trainimages, trainlabels, testimages, testlabels = loadidximages(filepaths["train_images"]), loadidxlabels(filepaths["train_labels"]), loadidximages(filepaths["test_images"]), loadidxlabels(filepaths["test_labels"])
    images, labels = np.vstack((trainimages, testimages)), np.concatenate((trainlabels, testlabels))

    rotatedimages = []
    for img in images:
        flipped = np.fliplr(img)
        rotated = np.rot90(flipped)
        rotatedimages.append(rotated)
    
    return np.array(rotatedimages), labels

def resizeimages(images, size=(16, 16)):
    resizedimages = []
    
    for img in images:
        pilimg = Image.fromarray(img)
        resizedimg = pilimg.resize(size, Image.LANCZOS)
        recolouredimg = PIL.ImageOps.invert(resizedimg)
        resizedimages.append(np.array(recolouredimg))
    
    return np.array(resizedimages)

def getlabelmapping():
    mapping = {
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
        30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
    }
    return mapping

def split_dataset(images, labels, learnsamplesperclass, testsamplesperclass):
    trainidx, testidx = [], []
    labelmapping = getlabelmapping()
    
    for labelval in labelmapping.keys():
        classindices = np.where(labels == labelval)[0]
        totalsamples = len(classindices)
        availablesamples = min(totalsamples, learnsamplesperclass + testsamplesperclass)
        
        if availablesamples < (learnsamplesperclass + testsamplesperclass):
            testcount = min(testsamplesperclass, int(availablesamples * 0.3))
            learncount = availablesamples - testcount
        else:
            learncount, testcount = learnsamplesperclass, testsamplesperclass

        random.seed(42 + labelval)
        selectedindices = random.sample(list(classindices), learncount + testcount)
        learnindices, testindices = selectedindices[:learncount], selectedindices[learncount:]
        
        trainidx.extend(learnindices)
        testidx.extend(testindices)
        
    return trainidx, testidx

def save_dataset(images, labels, indices, directory, labelmapping):
    for labelval in labelmapping.keys():
        classdir = os.path.join(directory, labelmapping[labelval])
        os.makedirs(classdir, exist_ok=True)
        classindices = [idx for idx in indices if labels[idx] == labelval]
        
        for i, idx in enumerate(classindices):
            img = Image.fromarray(images[idx])
            img.save(os.path.join(classdir, f"{labelmapping[labelval]}_{i}.png"))

def main():
    learndir, testdir = "learn", "test"
    learnsamplesperclass, testsamplesperclass = 3200, 4

    images, labels = loademnistbyclass()
    images, labelmapping = resizeimages(images, size=(8, 8)), getlabelmapping()
    trainindices, testindices = split_dataset(images, labels, learnsamplesperclass, testsamplesperclass)

    os.makedirs(learndir, exist_ok=True)
    os.makedirs(testdir, exist_ok=True)

    save_dataset(images, labels, trainindices, learndir, labelmapping)
    save_dataset(images, labels, testindices, testdir, labelmapping)

if __name__ == "__main__":
    main()