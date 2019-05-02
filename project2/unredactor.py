from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.metrics import accuracy_score
import argparse
import numpy as np
import os
import pickle

def bag_of_words(text, dictionary):
    """
    Convert a given snippet of text into a bag of words format, 
    returning a dict from word to count.
    """
    text = text.lower()
    words = word_tokenize(text)
    counter = Counter(words)
    output = {
        dictionary[word]: count 
        for word, count in counter.items() 
        # This gets rid of stop words as well.
        if word in dictionary
    }
    return output

def get_dictionary(dict_file):
    """
    Reads the dictionary. This simply contains a list of words.
    We filter out stop words.
    """
    stop = set(stopwords.words('english'))
    data = {}
    index = 0
    for line in dict_file:
        word = line.strip().lower()
        if len(word) > 0 and not word in stop:
            data[word] = index
            index += 1
    return data

def return_255():
    return 255

def read_training_files(files, dictionary, name_ids=None):
    """
    input: a list of files in the format this supports
    output: a sparse matrix with 1 row per redacted word. Each row of
    the matrix is of the form [# of redacted letters, bow of text],
    so column 1 is dense, while all other columns are highly sparse.
    """
    # First pass through the files, we merely read the number at the top
    # of the file and get a total size we are going to need Then we
    # allocate that size.
    total_samples = 0
    for f in files:
        with open(f, 'r') as example:
            total_samples += int(example.readline())
    print("Training on %d redacted names across %d comments" %(
            total_samples, len(files)
    ))
    # Create a matrix for holding all of our data and incrementally
    # update it.  use int16 for memory purposes.
    m = sparse.dok_matrix((total_samples, len(dictionary) + 1), dtype='int16')
    if name_ids is None:
        # Then we are in charge of reading the full set of names and deciding
        # what class it will belong to.  We do this by counting the top 255
        # most popular names - all other names get an id indicating other.
        name_counter = Counter()
        for f in files:
            with open(f, 'r') as input_file:
                num_redacted_words = int(input_file.readline())
                for i in range(num_redacted_words):
                    name = input_file.readline().strip()
                    name_counter[name] += 1
        num_names = 255
        names = [name for name, _ in name_counter.most_common(num_names - 1)]
        # And use those to populate a default dict:
        name_ids = defaultdict(return_255)
        for i, name in enumerate(names):
            name_ids[name] = i

    # Do it again, but now, actually parse everything and stick it in
    # the matrix
    answers = []
    example_index = 0
    # And in round two, we parse the actual data
    for f in files:
        with open(f, 'r') as input_file:
            num_redacted_words = int(input_file.readline())
            # Skip to the meat
            lens = []  # lengths of each of the redacted words
            for i in range(num_redacted_words):
                name = input_file.readline().strip()
                answers.append(name_ids[name])
                lens.append( len(name) )
            text = input_file.read()
            bow = bag_of_words(text, dictionary)
            for i in range(num_redacted_words):
                # First col is the number of chars
                m[example_index + i, 0] =  lens[i]
                for word, count in bow.items():
                    m[example_index + i, word + 1] = count 
            example_index += num_redacted_words
    return m.tocoo(), np.array(answers), name_ids

def load(input_dir, name_ids=None):
    # There are two things in input_dir:
    # 1) A dictionary file
    # 2) A bunch of data
    dict_file = input_dir + '/dict.txt'
    with open(dict_file, 'r') as df:
        dictionary = get_dictionary(df)
    stash_dir = input_dir + '/stash'
    # Double check if stash exists - if, so we can just return info from
    # with in that.
    if not os.path.isdir(stash_dir):
        os.mkdir(stash_dir)
    # Check that all files exist:
    files = ['bow.npz', 'ground_truth.npy', 'name_map.pickle']
    exist = all([os.path.isfile(stash_dir + '/' + f) for f in files])
    if exist:
        print("Loading previously preprocessed data...")
        # Then load them
        bow = sparse.load_npz(stash_dir + '/bow.npz')
        ground_truth = np.load(stash_dir + '/ground_truth.npy')
        with open(stash_dir + '/name_map.pickle', 'rb') as nmf:
            name_ids = pickle.load(nmf)
        return bow, ground_truth, name_ids, dictionary
    # Otherwise, we need to actually generate the data.
    print("Preprocessing input comments...")
    # Generate the list of files
    files = [
        input_dir + '/' + f 
        for f in os.listdir(input_dir) 
        if f.endswith('.redacted')
    ]
    bow, ground_truth, name_ids = read_training_files(files, dictionary, name_ids)
    # And stash them for later.
    sparse.save_npz(stash_dir + '/bow.npz', bow)
    np.save(stash_dir + '/ground_truth.npy', ground_truth)
    with open(stash_dir + '/name_map.pickle', 'wb') as nmf:
        pickle.dump(name_ids, nmf)
    # Return
    return bow, ground_truth, name_ids, dictionary

def train(input_dir):
    # Now, we simply train the model against training_data and
    # ground_truth. This produces the model we are going to use. 
    training_data, ground_truth, name_ids, dictionary = load(input_dir)
    training_data = training_data.tocsr()
    print("Data is populated. Constructing Naive Bayes model")
    clf = ComplementNB()
    # For some reason, this is really bad at training with a single
    # large dataset, so we break it up into chunks of 1000 samples and
    # train each on individually.
    # classes = np.array(sorted(list(name_ids.values())))
    # num_samples = training_data.shape[0]
    # for i in range(0, num_samples, 1000):
        # start = i
        # end = min(i + 1000, num_samples)
        # print('Updating for for %d through %d' % (start, end))
        # clf.partial_fit(
            # training_data[start:end],
            # ground_truth[start:end],
            # classes=classes
        # )
    # Train only on 'other' data that is one of the most popular names:
    mask = ground_truth != 255
    clf.fit(training_data[mask], ground_truth[mask])
    return clf, dictionary, name_ids

def test(clf, name_ids, dictionary, test_dir):
    # Get the training_data in a matrix form
    # clf.predict(training_data)
    test_data, ground_truth, _, _ = load(test_dir, name_ids)
    test_data = test_data.tocsr()
    # Filter out 'other' data
    mask = ground_truth != 255
    test_data = test_data[mask]
    ground_truth = ground_truth[mask]
    print("Populated data from test directory. Applying model to find accuracy...")
    # Compare predictions to reality
    predicted_data = clf.predict(test_data)
    # We use this trick to ensure that 255 (the 'other') category
    # doesn't compare equal to itself. Probably could have used np.nan
    # here, but that's just nasty
    map_255_to_256 = lambda i: 256 if i == 255 else i
    compare_data = np.vectorize(map_255_to_256)(predicted_data)
    return accuracy_score(ground_truth, compare_data)

def main():
    parser = argparse.ArgumentParser(
        description='Train a model and test it against test data'
    )
    # Directory with input files and dictionary
    parser.add_argument('--input-dir', required=True)
    # Directory with test case files for reporting error rate.
    parser.add_argument('--test-dir', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    clf, dictionary, name_ids = train(input_dir)

    print("Training complete. Evaluating accuracy...")

    test_dir = args.test_dir
    accuracy = test(clf, name_ids, dictionary, test_dir)

    print("This test achieved %f%% accuracy"%(100 * accuracy))

if __name__ == '__main__':
    main()

