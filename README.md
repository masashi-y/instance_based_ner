Code for [Label-Agnostic Sequence Labeling by Copying Nearest Neighbors](https://www.aclweb.org/anthology/P19-1533.pdf).

The code is based on [the original implementation](https://github.com/swiseman/neighbor-tagging), but modified in a way I like.

The results on the NER task is not exactly the same with slightly better scores on the development set (prec. 0.94918 / rec.: 0.95880 / F1: 0.95397), but the training proceeds similarly.



# Dependencies
The code is tested on Python 3.8.2 and libraries in `requirements.txt`.

# Training
To train the neighbor-based NER model, run:
```
python train_words.py cuda=0 save=mynermodel.pt
```

For available options, see `config.yaml` or:
```
python train_words.py --help
```

By default, it uses the train and validation splits of CoNLL2003 downloaded using [huggingface/datasets](https://github.com/huggingface/datasets).

You can specify to use test set as follows:
```
python train_words.py validation_split=test
```

If you have your own datasets, you can do:
```
python train_words.py train_split='[/path/to/train.words /path/to/train.tags]' validation_split='[/path/to/dev.words, /path/to/dev.tags]'
```

where each input file lists one example sentence/tag sequence per line.

# Evaluation
To evaluate on the development set, run:

```
python train_words.py eval=true cuda=0 trained_weights=mynermodel.pt
```

You can omit `trained_weights` to use pretrained (not finetuned) contextualized embeddings e.g., BERT.


For transfer evaluation:
```
python train_words.py trained_weights=mynermodel.pt validation_split='[data/onto/dev.words, data/onto/dev.ner]'
```

You can find those files in the original repo if you want:)
