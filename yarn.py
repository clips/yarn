import numpy as np
import re

from collections import defaultdict
from itertools import chain
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from reach import Reach


def calc_accuracy(output):
    """
    Calculates an accuracy score for the output of yarn.

    :param output: A dictionary of tuples representing the output of the system.
    :return: An accuracy score between 0 and 1.
    """

    total = list(chain.from_iterable(output.values()))
    return len([x for x, y in total if x == y]) / len(total)


def dict_to_tuple(dictionary):
    """
    Helper function.
    Creates a list of tuples from a dict of lists

    :param dictionary:
    :return: A list of tuples.
    """

    labels = []
    vectors = []

    for k, v in dictionary.items():
        for vec in v:
            labels.append(k)
            vectors.append(vec)

    return labels, vectors


def compute_nearest_neighbours(definitions, abstracts):
    """
    Computes nearest neighbours from abstracts to definitions.

    :param definitions: A dictionary of dictionaries containing vectors.
    The top key is the Ambiguous term, the bottom key is the CUI.

        Example: {AMBIGTERM: {CUI1: VECTOR, CUI2: VECTOR}}

    :param abstracts: Like definitions.
    :return: A dictionary, the keys of which are the ambiguous terms, and the values are lists of tuples.
    The first item of each tuple is the true class, the second item of each tuple is the predicted class.

        example: {AMBIGTERM1: [(y1, y_pred1), (y2, y_pred2), ...]}
    """

    output = {}

    for k, v in abstracts.items():

        results = []

        labels, vectors = dict_to_tuple(v)

        try:
            targets, matrix = dict_to_tuple(definitions[k])
        except KeyError:
            continue

        matrix = np.array([Reach.normalize(x) for x in matrix])

        for vec in vectors:

            vec = Reach.normalize(vec)
            result = -vec.dot(matrix.T)
            results.append(targets[np.argsort(result)[0]])

        output[k] = list(zip(labels, results))

    return output


def prep_run(yarn, definitions, abstracts, windowsize, focus_word=False):
    """
    Runs YARN on some definitions and some abstracts, with a pre-specified window size.

    :param yarn: The instance of yarn to use.
    :param definitions: A dictionary of dictionaries, representing the definitions.
    The top key is the ambiguous term, the bottom key the CUI.
    The definitions themselves should be tokenized strings, i.e. not lists of words.

        example: {AMBIGTERM1: {CUI1: [definition1, ...], CUI2: [...]},
                  AMBIGTERM2: ...,
                  AMBIGTERMn: ...}

    :param abstracts: A dictionary of dictionaries, representing the abstracts.
    Structure is similar to definitions, above.

    :param windowsize: The size of the windows to which to truncate each abstract.
    :param focus_word: Whether to include the focus word in the window.
    :return: A dictionary, the keys of which are the ambiguous terms, and the values are lists of tuples.
    The first item of each tuple is the true class, the second item of each tuple is the predicted class.

        example: {AMBIGTERM1: [(y1, y_pred1), (y2, y_pred2), ...]}
    """
    transformed_definitions = defaultdict(dict)
    transformed_abstracts = defaultdict(dict)

    # Create the definition vectors.
    for k, v in definitions.items():
        for cui, texts in v.items():
            transformed_definitions[k.lower()][cui] = [yarn.vectorize_no_window(texts)]

    # Create the windows for the abstracts and vectorize them.
    for k, v in abstracts.items():
        for cui, texts in v.items():
            transformed_abstracts[k.lower()][cui] = list(yarn.vectorize_window(texts, k.lower(), window_size=windowsize, include_focus=focus_word))

    # Compare the representations
    output = compute_nearest_neighbours(dict(transformed_definitions), dict(transformed_abstracts))

    return output


class Yarn:

    def __init__(self, vectorizer, composer_1=np.sum, composer_2=np.sum, combiner=np.mean):
        """
        Class for running the YARN experiments.

        :param vectorizer: an instance of Reach, loaded with your favorite representations.
        :param composer_1: The composition function used on the definitions.
        :param composer_2: The composition function used on the abstracts
        :param combiner: The composition function used to combined the first-order vectors into second-order vectors.
        :return: None
        """

        self.vectorizer = vectorizer
        self.composer_1 = composer_1
        self.composer_2 = composer_2
        self.combiner = combiner

        # Removes all punctuation
        self.remover = re.compile("\W")

    def vectorize_no_window(self, texts):
        """
        Vectorizes an entire corpus of texts. Highly similar to Vectorize_window, below, but without the windows.

        :param texts: A list of texts.
        :return: A single vector, representing the combination of all input texts.
        """

        transformed = []

        for t in texts:

            # Remove all punctuation, lower the text and split it into words.
            t = self.remover.sub(" ", t.lower()).split()

            # Compose the word vectors for a given text into a single representation.
            vector = self.composer_1(self.vectorizer.vectorize(t), axis=0)
            transformed.append(vector)

        # There should be as many transformed texts as original texts.
        assert len(transformed) == len(texts)

        # Combines all transformed representations into a single representation.
        return self.combiner(transformed, axis=0)

    def vectorize_window(self, texts, focus_word, window_size, include_focus=False):
        """
        Creates windows of a specified window size around a term, and vectorizes and combines these windows.
        Removes all punctuation.

        Example:
        window size = 2.
        focus word = cancer.

            "The patient has cancer , she will not survive" -> "patient has she will"

        :param texts: A list of texts.
        :param focus_word: The term around which to create the windows.
        :param window_size: The size of the window.
        :param include_focus: Whether to include the focus word in the window itself. Currently, this has a
        detrimental effect on performance.
        :return: A vector, representing the combined windows.
        """

        transformed = []

        for t in texts:

            # Remove all non-alphanumeric characters.
            t = self.remover.sub(" ", t.lower()).split()
            # Do the same thing to the focus word.
            focus = self.remover.sub(" ", focus_word).split()

            # Remove all stop words, except when the stop word is a part of the focus word.
            t = [x for x in t if x not in ENGLISH_STOP_WORDS or x == focus_word]

            intermediate = []

            for idx, w in enumerate(t):

                if w in focus:

                    # Create the window
                    # From 0 or idx - window size to the focus word.
                    window = t[max(0, idx - window_size):idx]
                    if include_focus:
                        # Add the focus word
                        window.extend(focus)
                    # From the focus word to focus word + window size
                    window.extend(t[idx + 1:idx + window_size + 1])

                    # Compose the window into a representation.
                    intermediate.append(self.composer_2(self.vectorizer.vectorize(window), axis=0))

            # Combine all composed windows into a single representation.
            transformed.append(self.combiner(intermediate, axis=0))

        return transformed


if __name__ == "__main__":

    import logging
    import time
    import json

    # Setup
    logging.basicConfig(level=logging.INFO)

    umls = "sample_data/umls_sample.json"
    msh = "sample_data/abstracts_example.json"
    path_to_embeddings = ""
    use_subset = False

    embeddings = Reach(path_to_embeddings, header=True, verbose=False)
    
    logging.info("loaded embeddings.")

    start = time.time()

    y = Yarn(embeddings)

    umls = json.load(open(umls))
    msh = json.load(open(msh))

    if use_subset:

        subset = [u'di',
                  u'tat',
                  u'erp',
                  u'ori',
                  u'crna',
                  u'pep',
                  u'de',
                  u'hip',
                  u'glycoside',
                  u'sterilization',
                  u'ra',
                  u'don',
                  u'ecg',
                  u'cell',
                  u'cholera',
                  u'lactation',
                  u'rdna',
                  u'synapsis',
                  u'aa',
                  u'ion']

        msh = {k: v for k, v in msh.items() if k.lower() in subset}

    results = prep_run(y, umls, msh, windowsize=6)
    score = calc_accuracy(results)
