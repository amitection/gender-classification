import random

from src.dataloader.loader import DatasetLoader
from src.classifiers.NaiveBayesClassifier import NaiveBayesClassfier
from src.dataloader.preprocess import PreProcessor

DATA_DIR = '../resources/dataset'

if __name__ == '__main__':
    pp = PreProcessor()

    # Comment the below line if data already generated
    #pp.generateWordFreqByGender(pp.extractAndSeperateGenderWCList());

    dl = DatasetLoader()
    data, target = dl.get_data(DATA_DIR)

    avg_acc = 0;

    dataset = list(zip(data, target))

    for i in range(0,10):
        random.shuffle(dataset)

        data1, target1 = zip(*dataset)

        classfier = NaiveBayesClassfier()
        classfier.fit(data1[:30], target1[:30])
        pred = classfier.predict(data1[31:])
        true = target1[31:]

        accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
        print("\nAccuracy: {0:.4f}".format(accuracy))

        avg_acc += accuracy


    print("Average Accuracy: {0:.4f}".format(avg_acc/10))