__author__ = 'haseeb'
import pandas as pn
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

TRAIN_FILE = "../data/labeledTrainData.tsv"


def get_train_data(train_file):
    return pn.read_csv(train_file, header=0, delimiter="\t", quoting=3)


def clean_date(data):
    cleaned_data = []
    (number_enteries, colums) = data.shape
    for entery_id in range(number_enteries):
        cleaned_data.append(clean_review(data["review"][entery_id]))
        if (entery_id+1) % 1000 == 0:
            print str(entery_id+1) + "/" + str(number_enteries) + " has been processed"
    return cleaned_data


def clean_review(review):
    text = BeautifulSoup(review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    stops = stopwords.words("english") #faster
    words = [w for w in words if w not in stops]
    return " ".join(words)


if __name__ == '__main__':
    train_data = get_train_data(TRAIN_FILE)
    print len(clean_date(train_data))