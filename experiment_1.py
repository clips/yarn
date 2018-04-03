"""Reproduce the experiment from the paper."""
from reach import Reach
from yarn import Yarn, prep_run, calc_accuracy

if __name__ == "__main__":

    import logging
    import time
    import json

    # Setup
    # logging.basicConfig(level=logging.INFO)

    umls = "sample_data/umls_sample.json"
    msh = "sample_data/abstracts_example.json"
    path_to_embeddings = ""
    use_subset = False

    # Be sure to set add_unk to True, or to mark the UNK index.
    embeddings = Reach.load(path_to_embeddings, header=True, unk_word="UNK")

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
