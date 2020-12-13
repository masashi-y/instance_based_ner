Code for [Label-Agnostic Sequence Labeling by Copying Nearest Neighbors](https://www.aclweb.org/anthology/P19-1533.pdf).

# Dependencies
The code was developed and tested with pytorch-pretrained-bert 0.5.1. (So you may need to do something like `pip install pytorch-pretrained-bert==0.5.1`).

# Training
To train the neighbor-based NER model, run:
```
python train_words.py cuda=0 detach_db=true save=mynermodel.pt
```

By default the above script will save a database file to the argument of `-db_fi`. Once the database file has been saved, you can rerun the above with `-load_saved_db` to avoid retokenizing everything.

The other models can be trained analogously, by substituting in the correct data files in `data/`; see the options in `train_words.py`. For POS tasks, the `-acc_eval` flag should be used, and we used a batch size of 20 and a learning rate of 2E-05.

# Evaluation
To evaluate on the development set, run:
```
python train_words.py cuda=0 pretrained=mynermodel.pt max_num_neighbors=100 pred_shard_size=64 -just_eval dev
```

To run evaluation with recomputed neighbors (under the fine-tuned, rather than pretrained BERT), use the option `-just_eval dev-newne`.

You can transfer evaluation as follows:
```
python -u train_words.py -cuda -pretrained mynermodel.pt max_num_neighbors=100 pred_shard_size=64 -just_eval "dev-newne" zero_shot=true
```
