import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__file__)


EmbeddingFun = Callable[[torch.FloatTensor], torch.FloatTensor]


@dataclass
class Alignment:
    start_index: int
    end_index: int


@dataclass
class Instance:
    words: List[str]
    alignments: List[Alignment]
    wordpieces: List[int]
    tags: List[str]

    def __len__(self):
        return len(self.words)


@dataclass
class Batch:
    instances: List[Instance]
    neighbor_instances: List[Instance]
    targets: torch.sparse.LongTensor
    x_mapper_sparse: torch.sparse.LongTensor
    neighbor_mapper_sparse: torch.sparse.LongTensor
    ignore_index: int


@dataclass
class RandomBatch:
    start_index: int
    end_index: int


def align_wordpieces(
    words: List[str], wordpieces: List[str], lower: bool = False
) -> List[Alignment]:
    """
    maps each word idx to start and end idx w/ in wordpieces.
    assumes wordpieces is padded on either end with CLS and SEP
    """
    results = []
    curr_start, word_index = 1, 0  # start at 1, b/c of CLS
    buffer_str = ""
    for wordpiece_index, wordpiece in enumerate(
        wordpieces[1:-1], 1
    ):  # ignore [SEP] final token

        if wordpiece.startswith("##"):
            wordpiece = wordpiece[2:]

        buffer_str += wordpiece

        word = words[word_index]
        if lower:
            word = word.lower()

        if buffer_str == word or buffer_str == "[UNK]":
            results.append(
                Alignment(start_index=curr_start, end_index=wordpiece_index + 1)
            )
            curr_start = wordpiece_index + 1
            word_index += 1
            buffer_str = ""

    assert word_index == len(words)
    return results


def _iter_minibatches(
    instances: List[Instance], batch_size: int
) -> Iterator[Tuple[List[Instance], Tuple[int, int]]]:
    buffer_ = []
    curr_len = len(instances[0])
    start = 0
    for batch_index, instance in enumerate(instances):
        if len(instance) != curr_len or batch_index - start == batch_size:
            yield buffer_, (start, batch_index)
            buffer_ = []
            curr_len = len(instance)
            start = batch_index
        else:
            buffer_.append(instance)
    if buffer_:
        yield buffer_, (start, len(instances))


def _get_wordpiece_to_word_mapper(
    batch_instances: List[Instance], align_strategy: str = "first",
) -> torch.sparse.LongTensor:
    """
    calculate word to word piece alignment for everybody:
    x_mapper will be batch_size x seq_len x max_batch_wrdpieces
    neighbor_mapper will be num_neighbors x max_ne_len x max_ne_wrdpieces
    """

    indices = []
    for batch_index, instance in enumerate(batch_instances):
        for word_index, (first_wordpiece_index, last_wordpiece_index) in enumerate(
            instance.alignments
        ):

            if align_strategy == "sum":
                indices.extend(
                    [
                        (batch_index, word_index, wordpiece_index)
                        for wordpiece_index in range(
                            first_wordpiece_index, last_wordpiece_index
                        )
                    ]
                )
            elif align_strategy == "first":
                indices.append((batch_index, word_index, first_wordpiece_index))
            elif align_strategy == "last":
                indices.append((batch_index, word_index, last_wordpiece_index - 1))
            else:
                raise KeyError(
                    f'alignment strategy "{align_strategy}" is not supported'
                )

    batch_size = len(batch_instances)
    max_wordpieces = max(len(instance.wordpieces) for instance in batch_instances)
    max_words = max(len(instance) for instance in batch_instances)

    values = torch.ones(len(indices), dtype=torch.long)
    return torch.sparse_coo_tensor(
        list(zip(*indices)), values, (batch_size, max_words, max_wordpieces)
    )


def get_all_embeddings(
    instances: List[Instance],
    batch_size: int,
    embedding_fun: EmbeddingFun,
    device: torch.device,
) -> torch.FloatTensor:

    results = []
    for batch_instances, _ in _iter_minibatches(instances, batch_size):
        # batch_size x max_wordpieces
        batch = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(instance.wordpieces) for instance in batch_instances],
            padding_value=0,
            batch_first=True,
        )
        results.append(embedding_fun(batch.to(device)))
    return torch.cat(results, dim=0)


class CoNLL2003Dataset(object):

    align_strategy_choices = ["sum", "first", "last"]

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        align_strategy: str = "last",
        subsample: int = 2500,
        max_num_neighbors: int = 50,
        neighbor_candidate_instances: Optional[List[Instance]] = None,
        random_neighbors_in_train: bool = False,
    ):

        assert split in ["train", "dev", "test"]

        self.validation = split != "train"
        assert (
            align_strategy in self.align_strategy_choices
        ), f"unsupported alignment strategy: {align_strategy}"

        assert not (
            self.validation and random_neighbors_in_train
        ), "`random_neighbors_in_train` can not be set True on validation set"

        assert (
            not self.validation or neighbor_candidate_instances is not None
        ), "`neighbor_candidate_instances` must be set on validation set"

        self.align_strategy = align_strategy
        self.subsample = subsample
        self.tokenizer = tokenizer
        self.max_num_neighbors = max_num_neighbors
        self.random_neighbors_in_train = random_neighbors_in_train

        self.instances = self._load_dataset(split)

        if self.validation:
            self.neighbor_candidate_instances = neighbor_candidate_instances
        else:
            self.neighbor_candidate_instances = self.instances

        self.tag_to_neighbor_indices = defaultdict(set)
        for sent_index, instance in enumerate(self.neighbor_candidate_instances):
            for tag in instance.tags:
                self.tag_to_neighbor_indices[tag].add(sent_index)

        self.tags = sorted(
            {
                tag
                for instance in self.neighbor_candidate_instances
                for tag in instance.tags
            }
        )
        self.minibatches = None
        self.top_neighbors = None

    def __len__(self) -> int:
        if self.minibatches is None:
            return 0
        return len(self.minibatches)

    def _load_dataset(self, split: str) -> List[Instance]:

        dataset = load_dataset("conll2003", split=split)
        results = []
        for sample in dataset:
            words = sample["tokens"]
            tags = [
                dataset.features["ner_tags"].feature.int2str(tag_id)
                for tag_id in sample["ner_tags"]
            ]
            wordpieces = (
                ["[CLS]"] + self.tokenizer.tokenize(" ".join(words)) + ["[SEP]"]
            )
            alignments = align_wordpieces(
                words, wordpieces, lower=self.tokenizer.do_lower_case
            )
            wordpiece_indices = self.tokenizer.convert_tokens_to_ids(wordpieces)
            results.append(
                Instance(
                    words=words,
                    alignments=alignments,
                    wordpieces=wordpiece_indices,
                    tags=tags,
                )
            )

        # shuffle before sorting by length
        random.shuffle(results)
        results.sort(key=lambda instance: len(instance))
        return results

    def compute_top_neighbors(
        self,
        batch_size: int,
        embedding_fun: EmbeddingFun,
        topk: int,
        device: torch.device,
        cosine: bool = True,
    ) -> None:

        instances_embeddings = get_all_embeddings(
            self.instances, batch_size, embedding_fun, device
        )
        if cosine:
            instances_embeddings = torch.nn.functional.normalize(
                instances_embeddings, p=2, dim=1
            )

        if self.validation:
            candidates_embeddings = get_all_embeddings(
                self.neighbor_candidate_instances,
                batch_size,
                embedding_fun,
                device=device,
            )
            if cosine:
                candidates_embeddings = torch.nn.functional.normalize(
                    candidates_embeddings, p=2, dim=1
                )
        else:
            candidates_embeddings = instances_embeddings

        similarities = instances_embeddings.mm(candidates_embeddings.t())
        similarities[
            torch.eye(similarities.size(0), dtype=torch.bool)
        ] = 0.0  # set diagonal to zero

        _, results = torch.topk(similarities, topk, dim=1)
        self.top_neighbors = [row.tolist() for row in results]

    def iter_batches(
        self,
        shuffle: bool = False,
        padding_value: int = 0,
        sinple_prediction: bool = False,
    ):
        if sinple_prediction:
            steps = list(range(len(self.minibatches)))
        else:
            steps = list(range(len(self)))

        if shuffle:
            random.shuffle(steps)

        for step in steps:
            yield self.get_batch(
                step, padding_value=padding_value, sinple_prediction=sinple_prediction
            )

    def get_batch(
        self,
        batch_or_sent_index: int,
        padding_value: int = 0,
        sinple_prediction: bool = False,
    ):
        """
        make a validation minibatch for actually predicting.
        N.B. only uses validation minibatches
        """

        if self.validation:
            if not sinple_prediction:
                batch = self.minibatches[batch_or_sent_index]
            else:
                batch = self._compute_batch(
                    batch_or_sent_index, batch_or_sent_index + 1
                )
        else:
            batch = self.minibatches[batch_or_sent_index]

            if isinstance(batch, RandomBatch):

                batch = self._compute_batch(
                    batch.start_index, batch.end_index, random_neighbors=True,
                )

        # batch_size x max_wordpieces
        x = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(instance.wordpieces) for instance in batch.instances],
            padding_value=padding_value,
            batch_first=True,
        )

        neighbors = torch.nn.utils.rnn.pad_sequence(
            [
                torch.LongTensor(instance.wordpieces)
                for instance in batch.neighbor_instances
            ],
            padding_value=padding_value,
            batch_first=True,
        )

        x_mapper = batch.x_mapper_sparse.to_dense()
        neighbor_mapper = batch.neighbor_mapper_sparse.to_dense()

        if not self.validation:

            sparse_targets = batch.targets.coalesce()
            mask = (
                torch.sparse_coo_tensor(
                    sparse_targets.indices(),
                    torch.ones_like(sparse_targets.values()),
                    sparse_targets.size(),
                )
                .to_dense()
                .bool()
            )
            targets = sparse_targets.to_dense()
            targets[~mask] = batch.ignore_index

            return x, neighbors, x_mapper, neighbor_mapper, targets

        gold_tags = [instance.tags for instance in batch.instances]

        tag_to_neighbors = self._make_tag_to_neighbors_dict(
            batch.neighbor_instances,
            subsample=not sinple_prediction,  # subsample when batch prediction
        )

        num_neighbors = len(batch.neighbor_instances)
        max_neighbor_seq_len = max(
            len(instance.tags) for instance in batch.neighbor_instances
        )

        tag_to_mask = []
        for tag in self.tags:
            indices = tag_to_neighbors[tag]
            values = torch.ones(len(indices), dtype=torch.float)
            mask = (
                torch.sparse_coo_tensor(
                    list(zip(*indices)), values, (num_neighbors, max_neighbor_seq_len)
                )
                .to_dense()
                .log()
            )
            tag_to_mask.append((tag, mask.unsqueeze(dim=1)))

        return x, neighbors, x_mapper, neighbor_mapper, tag_to_mask, gold_tags

    def _make_tag_to_neighbors_dict(
        self, neighbor_instances: List[Instance], subsample: bool = True
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        map each tag to location in neighbors
        """

        results = defaultdict(list)
        for sentence_index, instance in enumerate(neighbor_instances):
            for token_index, tag in enumerate(instance.tags):
                results[tag].append((sentence_index, token_index))

        if subsample:
            keys = {"O"}
            if self.subsample is not None:
                keys.update(results.keys())

            for key in keys:
                if key in results and len(results[key]) > self.subsample:
                    results[key] = random.sample(results[key], self.subsample)

        return results

    def _sample_neighbor_instances(
        self, start_index: int, end_index: int, random_neighbors: bool = False,
    ) -> List[Instance]:

        seen_indices = set() if self.validation else set(range(start_index, end_index))

        def sample(neighbors):
            if random_neighbors:
                return random.sample(neighbors, self.max_num_neighbors)
            return neighbors[: self.max_num_neighbors]

        neighbor_instances = [
            self.neighbor_candidate_instances[neighbor_index]
            for neighbor_index in sample(self.top_neighbors[start_index:end_index])
            if neighbor_index not in seen_indices
        ]

        # also add a neighbor for every missing tag
        neighbor_tags = {
            tag for instance in neighbor_instances for tag in instance.tags
        }

        for tag, candidate_indices in self.tag_to_neighbor_indices.items():
            if tag not in neighbor_tags:
                neighbor_instances.append(
                    random.choice(
                        [
                            self.neighbor_candidate_instances[candidate_index]
                            for candidate_index in candidate_indices
                            if candidate_index not in seen_indices
                        ],
                    )
                )

        return neighbor_instances

    def make_minibatches(self, batch_size: int):
        self.minibatches = []
        for _, (start_index, end_index) in _iter_minibatches(
            self.instances, batch_size
        ):
            if self.random_neighbors_in_train:
                batch = RandomBatch(start_index, end_index)
            else:
                batch = self._compute_batch(start_index, end_index)
            self.minibatches.append(batch)

    def _compute_batch(
        self, start_index: int, end_index: int, random_neighbors: bool = False,
    ) -> Batch:

        neighbor_instances = self._sample_neighbor_instances(
            start_index, end_index, random_neighbors=random_neighbors,
        )

        tag_to_neighbors = self._make_tag_to_neighbors_dict(
            neighbor_instances, subsample=True
        )

        instances = self.instances[start_index:end_index]

        batch_size = end_index - start_index
        max_seq_len = max(len(instance) for instance in instances)
        max_neighbor_seq_len = max(
            len(instance.tags) for instance in neighbor_instances
        )

        targets, target_indices = [], []
        max_correct = 0
        for batch_index, instance in enumerate(instances):
            for word_index in range(max_seq_len):

                corrects = [
                    sentence_index * max_neighbor_seq_len + token_index
                    for sentence_index, token_index in tag_to_neighbors[
                        instance.tags[word_index]
                    ]
                ]
                max_correct = max(max_correct, corrects)

                target_indices.extend(
                    [
                        (batch_index, word_index, correct_index)
                        for correct_index in range(len(corrects))
                    ]
                )
                targets.extend(corrects)

        targets = torch.sparse_coo_tensor(
            list(zip(*target_indices)), targets, (batch_size, max_seq_len, max_correct),
        )

        x_mapper_sparse = _get_wordpiece_to_word_mapper(
            instances, align_strategy=self.align_strategy,
        )

        neighbor_mapper_sparse = _get_wordpiece_to_word_mapper(
            neighbor_instances, align_strategy=self.align_strategy,
        )

        ignore_index = len(neighbor_instances) * max_neighbor_seq_len

        batch = Batch(
            instances,
            neighbor_instances,
            targets,
            x_mapper_sparse,
            neighbor_mapper_sparse,
            ignore_index,
        )

        return batch
