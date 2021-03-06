# yarn

Yarn is a system for creating vectorial concept representations from an ontology containing descriptions of these concepts. These concept representations can then be used to disambiguate terms, and link them to the appropriate concept. 

For more information, see the paper [_Using Distributed Representations to Disambiguate Biomedical and Clinical Concepts_](http://www.aclweb.org/anthology/W/W16/W16-29.pdf#page=89) by Stéphan Tulkens, Simon Šuster and Walter Daelemans, which was presented at the [BioNLP Workshop at ACL 2016](http://www.aclweb.org/aclwiki/index.php?title=BioNLP_Workshop).

# License

MIT

# Contributors

Stéphan Tulkens, Simon Suster, and Walter Daelemans.
If you use this work or build upon it, please cite our paper, as follows:

```
@inproceedings{tulkens2016using,
  title={Using Distributed Representations to Disambiguate Biomedical and Clinical Concepts},
  author={Tulkens, St{\'e}phan and {\v{S}}uster, Simon and Daelemans, Walter},
  booktitle={Proceedings of the 15th Workshop on Biomedical Natural Language Processing},
  pages={77--82},
  year={2016}
}
```

# Requirements

- Python 3
- Numpy
- [Reach](https://github.com/stephantul/reach)

All are available from pip

# Usage

Yarn requires:
- A set of word vectors
- A set of concepts, with their descriptions
- A set of documents with their ambiguous terms marked

The word vectors we used can be downloaded from the [BioASQ website](http://bioasq.org/news/bioasq-releases-continuous-space-word-vectors-obtained-applying-word2vec-pubmed-abstracts).

If you want to replicate the original experiments, you need to adhere to the formats below. If you want to use Yarn for your own experiments, e.g. just creating concept representations, you can choose your own format.

## concepts

Concepts are represented by a top-level dictionary of terms, concepts that pertain to these terms, and a list of descriptions (strings), of these concepts.

```
{"term":
  {"concept id_1":
    [description_1,
     description_2,
     ...
     description_n]
  },
  {"concept_id_2":
    [description_1,
     description_2,
     ...
     description_n]
  }
}
```

## documents

Similarly, documents to be disambiguated are represented by a dictionary. Note that each document _must_ contain at least one occurrence of the ambiguous term under which it is classified.

```
{"term":
  {"concept id_1":
    [document_1,
     document_2,
     ...
     document_n]
  },
  {"concept_id_2":
    [document_1,
     document_2,
     ...
     document_n]
  }
}
```

The original Yarn experiments were run with the MSH dataset ([Jimeno-Yepes 2011](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-223)) and the [2015AB](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html) release of the UMLS. Because these resources are not freely distributable, we were not able to redistribute them with this package.
