__author__ = 'haseeb'
import pandas as pn
import re
from bs4 import BeautifulSoup

TRAIN_FILE = "../data/labeledTrainData.tsv"


def get_train_data(train_file):
    return pn.read_csv(train_file, header=0, delimiter="\t", quoting=3)


def clean_date(data):
    (number_enteries, colums) = data.shape
    #for entery_id in range(number_enteries):
    for entery_id in range(10):
        text = BeautifulSoup(data["review"][entery_id]).get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", text)
        print letters_only


if __name__ == '__main__':
    train_data = get_train_data(TRAIN_FILE)
    print train_data.shape
    print train_data.columns.values
    clean_date(train_data)