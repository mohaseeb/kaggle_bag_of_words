__author__ = 'haseeb'
import pandas as pn
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

TRAIN_FILE = "../data/labeledTrainData.tsv"
TEST_FILE = "../data/testData.tsv"


def read_data(data_file):
    return pn.read_csv(data_file, header=0, delimiter="\t", quoting=3)


def clean_review(review):
    text = BeautifulSoup(review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    stops = stopwords.words("english") #faster
    words = [w for w in words if w not in stops]
    return " ".join(words)


def clean_date(data):
    cleaned_data = []
    number_enteries = len(data)
    for entery_id in range(number_enteries):
        cleaned_data.append(clean_review(data[entery_id]))
        if (entery_id+1) % 1000 == 0:
            print str(entery_id+1) + "/" + str(number_enteries) + " has been processed"
    return cleaned_data


def train(training_file):
    data = read_data(training_file)
    cleaned_data = clean_date(data["review"])
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, \
                                 max_features=5000)
    training_data = vectorizer.fit_transform(cleaned_data)
    training_data = training_data.toarray()
    print training_data.shape
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(training_data, data["sentiment"])
    return forest, vectorizer


def test(model, vectorizer):
    raw_test_data = read_data(TEST_FILE)
    print raw_test_data.columns.values
    cleaned_test_data = clean_date(raw_test_data["review"])
    test_data = vectorizer.transform(cleaned_test_data)
    test_data = test_data.toarray()

    result = model.predict(test_data)

    output = pn.DataFrame(data={"id": raw_test_data["id"], "sentiment": result})
    output.to_csv("../data/bag_of_words_results.csv", index=False, quoting=3)

if __name__ == '__main__':
    model, vectorizer = train(TRAIN_FILE)
    test(model, vectorizer)