__author__ = 'haseeb'
import pandas as pn
import re
import nltk.data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import logging
from gensim.models import word2vec
import numpy as np
from sklearn.cluster import KMeans
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
TRAIN_FILE = "../data/labeledTrainData.tsv"
TEST_FILE = "../data/testData.tsv"
TRAIN_UNLABELED_FILE = "../data/unlabeledTrainData.tsv"


def read_data(data_file):
    return pn.read_csv(data_file, header=0, delimiter="\t", quoting=3)


def review_to_words(review, remove_stop_words=True):
    text = BeautifulSoup(review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    if remove_stop_words:
        stops = stopwords.words("english")  # faster
        words = [w for w in words if w not in stops]
    return " ".join(words)


def review_to_wordslist(review, remove_stop_words=False):
    text = BeautifulSoup(review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split()
    if remove_stop_words:
        stops = set(stopwords.words("english"))  # faster
        words = [w for w in words if w not in stops]
    if len(words) == 0:
        print review
    return words


def review_to_sentence(review, tokenizer, remove_stop_words=False):
    raw_sentences = tokenizer.tokenize(review.decode('utf-8').strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordslist(raw_sentence, remove_stop_words))
    return sentences


def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)

    for word in words:
        if word in index2word_set:
            nwords += 1
            feature_vec = np.add(feature_vec, model[word])

    if nwords > 0.:
        feature_vec = np.divide(feature_vec, nwords)
    else:
        print nwords
    return feature_vec


def get_avg_feature_vecs(reviews, model, num_features):
    counter = 0.
    reviews_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if counter%1000 == 0:
            print "review %d of %d" % (counter, len(reviews))
        reviews_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter += 1
    return reviews_feature_vecs


def clean_date(data):
    cleaned_data = []
    number_enteries = len(data)
    for entery_id in range(number_enteries):
        cleaned_data.append(review_to_words(data[entery_id]))
        if (entery_id + 1) % 1000 == 0:
            print str(entery_id + 1) + "/" + str(number_enteries) + " has been processed"
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


def test_bag_of_words():
    model, vectorizer = train(TRAIN_FILE)
    test(model, vectorizer)


def build_word2vec_model(model_name, num_features):
    labeled_data = read_data(TRAIN_FILE)
    unlableled_data = read_data(TRAIN_UNLABELED_FILE)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentances = []
    for review in labeled_data["review"]:
        sentances += review_to_sentence(review, tokenizer)

    for review in unlableled_data["review"]:
        sentances += review_to_sentence(review, tokenizer)

    print len(sentances)

    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    model = word2vec.Word2Vec(sentances, workers=num_workers, size=num_features, min_count=min_word_count, window= \
        context, sample=downsampling)

    model.init_sims(replace=True)

    model.save(model_name)
    return model


def test_word2vec(build_model=True):
    labeled_data = read_data(TRAIN_FILE)
    unlableled_data = read_data(TRAIN_UNLABELED_FILE)
    test_data = read_data(TEST_FILE)

    num_features = 300
    model_name = "300features_40minwords_10context"

    if build_model:
        model = build_word2vec_model(model_name, num_features)
    else:
        model = word2vec.Word2Vec.load(model_name)

    print "creating average feature vecs for training reviews"
    clean_train_reviews = []
    for review in labeled_data["review"]:
        clean_train_reviews.append(review_to_wordslist(review, remove_stop_words=True))

    train_data_vecs = get_avg_feature_vecs(clean_train_reviews, model, num_features)

    print "creating average feature vecs for test reviews"
    clean_test_reviews = []
    for review in test_data["review"]:
        clean_test_reviews.append(review_to_wordslist(review, remove_stop_words=True))

    test_data_vecs = get_avg_feature_vecs(clean_test_reviews, model, num_features)

    forest = RandomForestClassifier(n_estimators=100)

    print "fitting a random forest to labeled training data ..."
    forest = forest.fit(train_data_vecs, labeled_data["sentiment"])

    result = forest.predict(test_data_vecs)

    output = pn.DataFrame(data={"id": test_data["id"], "sentiment": result})
    output.to_csv("../data/word2vec_averageVecs_result.csv", index=False, quoting=3)


def get_words_centeroid_map(model, num_clusters):
    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.index2word, idx ))

    # For the first 10 clusters
    for cluster in xrange(0,10):
        #
        # Print the cluster number
        print "\nCluster %d" % cluster
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
        print words

    return word_centroid_map


def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


def test_wort2vec_clustering(build_model=True):
    labeled_data = read_data(TRAIN_FILE)
    test_data = read_data(TEST_FILE)
    num_features = 300
    model_name = "300features_40minwords_10context"

    if build_model:
        model = build_word2vec_model(model_name, num_features)
    else:
        model = word2vec.Word2Vec.load(model_name)

    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / 5
    word_centroid_map = get_words_centeroid_map(model, num_clusters)

    clean_train_reviews = []
    for review in labeled_data["review"]:
        clean_train_reviews.append(review_to_wordslist(review, remove_stop_words=True))
    clean_test_reviews = []
    for review in test_data["review"]:
        clean_test_reviews.append(review_to_wordslist(review, remove_stop_words=True))

    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros((labeled_data["review"].size, num_clusters), dtype="float32")

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # Repeat for test reviews
    test_centroids = np.zeros((test_data["review"].size, num_clusters), dtype="float32")

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # Fit a random forest and extract predictions
    forest = RandomForestClassifier(n_estimators=100)

    # Fitting the forest may take a few minutes
    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(train_centroids, labeled_data["sentiment"])
    result = forest.predict(test_centroids)

    # Write the test results
    output = pn.DataFrame(data={"id": test_data["id"], "sentiment": result})
    output.to_csv("../data/word2vec_BagOfCentroids.csv", index=False, quoting=3)


if __name__ == '__main__':
    # test_bag_of_words()
    #test_word2vec(False)
    test_wort2vec_clustering(False)
