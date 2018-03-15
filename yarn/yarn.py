"""Experiments for the Yarn paper."""
import numpy as np
import re

from collections import defaultdict
from itertools import chain
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from reach import Reach


def calc_accuracy(output):
    """
    Calculate an accuracy score for the output of yarn.

    Parameters
    ----------
    output : dict
        A dictionary of tuples representing the output of the system.

    Returns
    -------
    accuracy : float
        An accuracy score between 0 and 1.

    """
    total = list(chain.from_iterable(output.values()))
    return len([x for x, y in total if x == y]) / len(total)


def dict_to_tuple(dictionary):
    """Create a list of tuples from a dict of lists."""
    labels = []
    vectors = []

    for k, v in dictionary.items():
        for vec in v:
            labels.append(k)
            vectors.append(vec)

    return labels, vectors


def compute_nearest_neighbours(definitions, abstracts):
    """
    Compute nearest neighbours from abstracts to definitions.

    Parameters
    ----------
    definitions : dictionary of dictionaries.
        A dictionary of dictionaries containing vectors.
        The top key is the Ambiguous term, the bottom key is the CUI.

            Example: {AMBIGTERM: {CUI1: VECTOR, CUI2: VECTOR}}
    abstracts : dictionary of dictionaries
        Like definitions.

    Returns
    -------
    result : dict
        A dictionary, the keys of which are the ambiguous terms, and the values
        are lists of tuples. The first item of each tuple is the true class,
        the second item of each tuple is the predicted class.

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

        matrix = Reach.normalize(np.asarray(matrix))
        vectors = Reach.normalize(np.asarray(vectors))

        for vec in vectors:

            result = -vec.dot(matrix.T)
            results.append(targets[np.argsort(result)[0]])

        output[k] = list(zip(labels, results))

    return output


def prep_run(yarn, definitions, abstracts, windowsize, focus_word=False):
    """
    Run YARN on some definitions and some abstracts.

    Parameters
    ----------
    yarn : Yarn
        The instance of yarn to use.
    definitions : dictionary of dictionaries
        A dictionary of dictionaries, representing the definitions.
        The top key is the ambiguous term, the bottom key the CUI.
        The definitions themselves should be tokenized strings, i.e. not lists
        of words.

        example: {AMBIGTERM1: {CUI1: [definition1, ...], CUI2: [...]},
                  AMBIGTERM2: ...,
                  AMBIGTERMn: ...}
    abstracts : dictionary of dictionaries
        A dictionary of dictionaries, representing the abstracts.
        Structure is the same as definitions, above.
    windowsize : int
        The size of the windows to which to truncate each abstract.
    focus_word : bool
        Whether to include the focus word in the window.

    Returns
    -------
    result : dict
        A dictionary, the keys of which are the ambiguous terms, and the values
        are lists of tuples. The first item of each tuple is the true class,
        the second item of each tuple is the predicted class.

        example: {AMBIGTERM1: [(y1, y_pred1), (y2, y_pred2), ...]}

    """
    t_definitions = defaultdict(dict)
    t_abstracts = defaultdict(dict)

    # Create the definition vectors.
    for k, v in definitions.items():
        k = k.lower()
        for cui, texts in v.items():
            t_definitions[k.lower()][cui] = [yarn.vectorize_no_window(texts)]

    # Create the windows for the abstracts and vectorize them.
    for k, v in abstracts.items():
        k = k.lower()
        for cui, texts in v.items():
            w = list(yarn.vectorize_window(texts,
                                           k,
                                           window_size=windowsize,
                                           include_focus=focus_word))
            t_abstracts[k.lower()][cui] = w

    # Compare the representations
    output = compute_nearest_neighbours(dict(t_definitions), dict(t_abstracts))

    return output


class Yarn(object):
    """
    Yarn is class for vectorizing and comparing strings of texts WSD.

    All the functions passed to the init must take an axis argument.

    Parameters
    ----------
    vectorizer : Reach
        An initialized Reach instance with word vectors.
    composer_1 : function, default np.sum
        A function which composes a matrix over its zeroth axis.
    composer_2 : function, default np.sum
        A function which composes a matrix over its zeroth axis.
    combiner : function, default np.mean
        A function which composes a matrix over its zeroth axis.

    """

    def __init__(self,
                 vectorizer,
                 composer_1=np.sum,
                 composer_2=np.sum,
                 combiner=np.mean):
        """Run yarn."""
        self.vectorizer = vectorizer
        self.composer_1 = composer_1
        self.composer_2 = composer_2
        self.combiner = combiner

        # Removes all punctuation
        self.remover = re.compile("\W")

    def vectorize_no_window(self, texts):
        """
        Vectorize an entire corpus of texts.

        Highly similar to Vectorize_window, below, but without the windows.
        The texts are assumed to be tokenized. Any punctuation and stop words
        will be removed.

        Parameters
        ----------
        texts : list
            A list of strings, where each string represents a text.

        Returns
        -------
        vec : np.array
            A single vector, representing the combination of all input texts.

        """
        transformed = []

        for t in texts:

            # Remove all punctuation, lower the text and split it into words.
            t = self.remover.sub(" ", t.lower()).split()

            # Compose the word vectors for a given text into a
            # single representation.
            vector = self.composer_1(self.vectorizer.vectorize(t), axis=0)
            transformed.append(vector)

        # There should be as many transformed texts as original texts.
        assert len(transformed) == len(texts)

        # Combines all transformed representations into a single
        # representation.
        return self.combiner(transformed, axis=0)

    def vectorize_window(self,
                         texts,
                         focus_word,
                         window_size,
                         include_focus=False):
        """
        Create windows around a term, vectorizes and combines these windows.

        This function removes all punctuation.

        Example:
        window size = 2.
        focus word = cancer.

            "The patient has cancer , she will not survive" ->
                        "patient has she will"

        Parameters
        ----------
        texts : list
            A list of strings, where each string represents a text.
        focus_word : list
            The term around which to create the windows. Focus word is a list
            because of multi-word units.
        window_size : int
            The size of the window used in vectorization.
        include_focus : bool
            Whether to include the focus word in the window itself.

        Returns
        -------
        comb : np.array
            A vector, representing the combined windows.

        """
        transformed = []

        for t in texts:

            # Remove all non-alphanumeric characters.
            t = self.remover.sub(" ", t.lower()).split()
            # Do the same thing to the focus word.
            focus = self.remover.sub(" ", focus_word).split()

            # Remove all stop words, except when the
            # stop word is a part of the focus word.
            t = [x for x in t if x not in ENGLISH_STOP_WORDS
                 or x in focus_word]

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
                    vectors = self.vectorizer.vectorize(window)
                    intermediate.append(self.composer_2(vectors, axis=0))

            # Combine all composed windows into a single representation.
            transformed.append(self.combiner(intermediate, axis=0))

        return transformed
