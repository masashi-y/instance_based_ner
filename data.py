import dataclasses
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

logger = logging.getLogger(__file__)


EmbeddingFun: Callable[[torch.FloatTensor], torch.FloatTensor]


@dataclasses.dataclass
class Alignment:
    start: int
    end: int


@dataclasses.dataclass
class Instance:
    words: List[str]
    alignments: List[Alignment]
    wordpieces: List[int]
    tags: List[str]

    def __len__(self):
        return len(self.words)


@dataclasses.dataclass
class Batch:
    start_index: int
    end_index: int
    neighbor_indices: List[int]
    targets: torch.sparse.LongTensor
    x_mapper_sparse: torch.sparse.LongTensor
    neighbor_mapper_sparse: torch.sparse.LongTensor


@dataclasses.dataclass
class RandomBatch:
    start_index: int
    end_index: int
    num_neighbors: int


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
            results.append(Alignment(start=curr_start, end=wordpiece_index + 1))
            curr_start = wordpiece_index + 1
            word_index += 1
            buffer_str = ""

    assert word_index == len(words)
    return results


class SentDB(object):

    align_strategy_choices = ["sum", "first", "last"]

    def __init__(
        self,
        train_sentences_file: str,
        train_tags_file: str,
        tokenizer: BertTokenizer,
        validation_sentences_file: str,
        validation_tags_file: str,
        lower: bool = False,
        align_strategy: str = "last",
        subsample: int = 2500,
    ):
        assert (
            align_strategy in self.align_strategy_choices
        ), f"unsupported alignment strategy: {align_strategy}"

        self.align_strategy = align_strategy
        self.subsample = subsample
        self.tokenizer = tokenizer
        self.lower = lower

        self.train_instances = self.process_words_tags_files(
            train_sentences_file, train_tags_file
        )
        self.validation_instances = self.process_words_tags_files(
            validation_sentences_file, validation_tags_file
        )

        self.tag_to_train_sent_index = defaultdict(set)
        for sent_index, instance in enumerate(self.train_instances):
            for tag in instance.tags:
                self.tag_to_train_sent_index[tag].add(sent_index)

        self.tags = sorted(
            {tag for instance in self.train_instances for tag in instance.tags}
        )
        self.train_minibatches = None
        self.validation_minibatches = None
        self.train_top_neighbors = None
        self.validation_top_neighbors = None

    def override_validation_with_test(
        self,
        test_words_file: str,
        test_tags_file: str,
        embedding_fun: EmbeddingFun,
        device: torch.device,
        batch_size: int = 128,
        topk: int = 50,  # 500,
    ):
        logger.info(
            "there were %d validation sentences", len(self.validation_instances)
        )

        self.validation_instances = self.process_words_tags_files(
            test_words_file, test_tags_file
        )

        self.validation_top_neighbors = None
        logger.info("now there are %d test sentences", len(self.validation_instances))
        logger.info("recomputing neighbors...")

        train_embeddings = self.get_all_embeddings(
            batch_size, embedding_fun, device, cosine=True
        )
        validation_embeddings = self.get_all_embeddings(
            batch_size, embedding_fun, device, cosine=True, validation=True
        )
        with torch.no_grad():
            similarities = validation_embeddings.mm(train_embeddings.t())
            _, results = torch.topk(similarities, topk, dim=1)
            self.validation_top_neighbors = [row.tolist() for row in results]

    def process_words_tags_files(
        self, words_file: str, tag_file: str
    ) -> List[Instance]:

        results = []
        with open(words_file) as f1, open(tag_file) as f2:
            for sent, tags in zip(f1, f2):
                sent = sent.strip()
                words = sent.split()
                tags = tags.strip().split()
                assert len(tags) == len(words)
                wordpieces = ["[CLS]"] + self.tokenizer.tokenize(sent) + ["[SEP]"]
                alignments = align_wordpieces(words, wordpieces, lower=self.lower)
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
        permutation = torch.randperm(len(results)).tolist()
        permutation.sort(key=lambda index: len(results[index]))
        return [results[index] for index in permutation]

    def get_all_embeddings(
        self,
        batch_size: int,
        embedding_fun: EmbeddingFun,
        device: torch.device,
        cosine: bool = True,
        validation: bool = False,
    ):
        if validation:
            instances = self.validation_instances
        else:
            instances = self.train_instances

        results = []

        for batch_index in range(0, len(instances), batch_size):
            # batch_size x max_wordpieces
            batch = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.LongTensor(instance.wordpieces)
                    for instance in instances[batch_index : batch_index + batch_size]
                ],
                padding_value=0,
                batch_first=True,
            )
            results.append(embedding_fun(batch.to(device)))

        results = torch.cat(results, dim=0)

        if cosine:
            results = torch.nn.functional.normalize(results, p=2, dim=1)

        return results

    def compute_top_neighbors(
        self,
        batch_size: int,
        embedding_fun: EmbeddingFun,
        topk: int,
        device: torch.device,
        cosine: bool = True,
        ignore_train: bool = False,
    ) -> None:

        train_embeddings = self.get_all_embeddings(
            batch_size, embedding_fun, device, cosine=cosine
        )

        if ignore_train:
            self.train_top_neighbors = None
        else:
            similarities = train_embeddings.mm(train_embeddings.t())
            # set diagonal to zero
            similarities[torch.eye(similarities.size(0), dtype=torch.bool)] = 0.0

            _, results = torch.topk(similarities, topk, dim=1)
            self.train_top_neighbors = [row.tolist() for row in results]

        validation_embeddings = self.get_all_embeddings(
            batch_size, embedding_fun, device, cosine=cosine, validation=True
        )

        similarities = validation_embeddings.mm(train_embeddings.t())
        _, results = torch.topk(similarities, topk, dim=1)
        self.validation_top_neighbors = [row.tolist() for row in results]

    def train_batch(self, batch_index: int, padding_value: int = 0):

        batch = self.train_minibatches[batch_index]

        if isinstance(batch, RandomBatch):

            batch = self.precompute_batch(
                batch.start_index,
                batch.end_index,
                batch.num_neighbors,
                random_neighbors=True,
                validation=False,
            )

        #  batch_size x max_wordpieces
        x = torch.nn.utils.rnn.pad_sequence(
            [
                torch.LongTensor(instance.wordpieces)
                for instance in self.train_instances[
                    batch.start_index : batch.end_index
                ]
            ],
            padding_value=padding_value,
            batch_first=True,
        )

        # num_neighbors x max_wordpieces
        neighbors = torch.nn.utils.rnn.pad_sequence(
            [
                torch.LongTensor(self.train_instances[neighbor_index].wordpieces)
                for neighbor_index in batch.neighbor_indices
            ],
            padding_value=padding_value,
            batch_first=True,
        )

        x_mapper = batch.x_mapper_sparse.to_dense()
        neighbor_mapper = batch.neighbor_mapper_sparse.to_dense()

        max_neighbor_seq_len = max(
            len(self.train_instances[neighbor_index].tags)
            for neighbor_index in batch.neighbor_indices
        )

        # neighb targets are in format num_neighbors*max_ne_len, so add one more option for ignore
        # TODO:aaaaaaaaaaaaaaaaaa
        ignore_index = len(batch.neighbor_indices) * max_neighbor_seq_len
        targets = batch.targets.to_dense()
        return x, neighbors, x_mapper, neighbor_mapper, targets

    def pred_batch(
        self,
        batch_or_sent_index: int,
        num_neighbors: int,
        padding_value: int = 0,
        batch: bool = True,
    ):
        """
        make a validation minibatch for actually predicting.
        N.B. only uses validation minibatches
        """

        if batch:
            batch = self.validation_minibatches[batch_or_sent_index]
            start_index, end_index = batch.start_index, batch.end_index
            x_mapper_sparse = batch.x_mapper_sparse
        else:
            start_index, end_index = batch_or_sent_index, batch_or_sent_index + 1
            x_mapper_sparse = self._get_wordpiece_to_word_mapper(
                [start_index], validation=True
            )

        # batch_size x max_wordpieces
        x = torch.nn.utils.rnn.pad_sequence(
            [
                torch.LongTensor(instance.wordpieces)
                for instance in self.validation_instances[start_index:end_index]
            ],
            padding_value=padding_value,
            batch_first=True,
        )

        neighbors = torch.nn.utils.rnn.pad_sequence(
            [
                torch.LongTensor(self.train_instances[neighbor_index].wordpieces)
                for neighbor_index in neighbor_indices
            ],
            padding_value=padding_value,
            batch_first=True,
        )

        x_mapper = x_mapper_sparse.to_dense()
        neighbor_mapper = self._get_wordpiece_to_word_mapper(
            neighbor_indices
        ).to_dense()

        gold_tags = [
            instance.tags
            for instance in self.validation_instances[start_index:end_index]
        ]

        neighbor_indices = self._compute_neighbor_indices(
            start_index,
            end_index,
            num_neighbors,
            random_neighbors=False,
            validation=True,
        )

        tag_to_neighbors = self._make_tag_to_neighbors_dict(
            neighbor_indices, subsample=batch  # subsample when batch prediction
        )

        num_neighbors = len(neighbor_indices)
        max_neighbor_seq_len = max(
            len(self.train_instances[neighbor_index].tags)
            for neighbor_index in neighbor_indices
        )

        tag_to_mask = {}
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
            tag_to_mask[tag] = mask.unsqueeze(dim=1)

        return x, neighbors, x_mapper, neighbor_mapper, tag_to_mask, gold_tags

    def _make_tag_to_neighbors_dict(
        self, neighbor_indices: List[int], subsample: bool = True
    ) -> Dict[str, List[Tuple[int, int]]]:

        # map each tag to location in neighbors
        tag_to_neighbors = defaultdict(list)
        for sentence_index, neighbor_index in enumerate(neighbor_indices):
            for token_index, tag in enumerate(
                self.train_instances[neighbor_index].tags
            ):
                tag_to_neighbors[tag].append((sentence_index, token_index))

        if subsample:
            keys = {"O"}
            if self.subsample is not None:
                keys.update(tag_to_neighbors.keys())

            for key in keys:
                if (
                    key in tag_to_neighbors
                    and len(tag_to_neighbors[key]) > self.subsample
                ):
                    tag_to_neighbors[key] = random.sample(
                        tag_to_neighbors[key], self.subsample
                    )

        return tag_to_neighbors

    def _compute_neighbor_indices(
        self,
        start_index: int,
        end_index: int,
        num_neighbors: int,
        random_neighbors: bool = False,
        validation: bool = False,
    ) -> List[int]:

        if validation:
            top_neighbors = self.validation_top_neighbors
            seen_indices = set()
        else:
            top_neighbors = self.train_top_neighbors
            seen_indices = set(range(start_index, end_index))

        def sample(neighbors):
            if random_neighbors:
                return random.sample(neighbors, num_neighbors)
            return neighbors[:num_neighbors]

        neighbor_indices = [
            neighbor_index
            for instance_index in range(start_index, end_index)
            for neighbor_index in sample(top_neighbors[instance_index])
            if neighbor_index not in seen_indices
        ]

        # also add a neighbor for every missing tag
        neighbor_tags = {
            tag
            for neighbor_index in neighbor_indices
            for tag in self.train_instances[neighbor_index].tags
        }

        for tag, sent_indices in self.tag_to_train_sent_index.items():
            if tag not in neighbor_tags:
                neighbor_indices.extend(
                    random.sample(
                        [
                            sent_index
                            for sent_index in sent_indices
                            if sent_index not in seen_indices
                        ],
                        1,
                    )
                )

        return neighbor_indices

    def make_minibatches(
        self,
        batch_size: int,
        num_neighbors: int,
        random_neighbors_in_train: bool = False,
        validation: bool = False,
    ):
        if validation:
            instances = self.validation_instances
            random_neighbors_in_train = False
        else:
            instances = self.train_instances

        minibatches = []
        for batch_index in range(0, len(instances), batch_size):
            if random_neighbors_in_train:
                batch = RandomBatch(
                    batch_index, batch_index + batch_size, num_neighbors
                )
            else:
                batch = self.precompute_batch(
                    batch_index,
                    batch_index + batch_size,
                    num_neighbors,
                    validation=validation,
                )
            minibatches.append(batch)

        if validation:
            self.validation_minibatches = minibatches
        else:
            self.train_minibatches = minibatches

    def _get_wordpiece_to_word_mapper(
        self, batch: List[int], validation: bool = False
    ) -> torch.sparse.LongTensor:
        """
        calculate word to word piece alignment for everybody:
        x_mapper will be batch_size x seq_len x max_batch_wrdpieces
        neighbor_mapper will be num_neighbors x max_ne_len x max_ne_wrdpieces
        """

        if validation:
            instances = self.validation_instances
        else:
            instances = self.train_instances

        indices = []
        for batch_index, instance_index in enumerate(batch):
            for word_index, (first_wordpiece_index, last_wordpiece_index) in enumerate(
                instances[instance_index].alignments
            ):
                if self.align_strategy == "sum":
                    indices.extend(
                        [
                            (batch_index, word_index, wordpiece_index)
                            for wordpiece_index in range(
                                first_wordpiece_index, last_wordpiece_index
                            )
                        ]
                    )
                elif self.align_strategy == "first":
                    indices.append((batch_index, word_index, first_wordpiece_index))
                elif self.align_strategy == "last":
                    indices.append((batch_index, word_index, last_wordpiece_index - 1))
                else:
                    raise KeyError(
                        f'alignment strategy "{self.align_strategy}" is not supported'
                    )

        batch_size = len(batch)
        max_wordpieces = max(len(instances[index].wordpieces) for index in batch)
        max_words = max(len(instances[index]) for index in batch)

        values = torch.ones(len(indices), dtype=torch.long)
        return torch.sparse_coo_tensor(
            list(zip(*indices)), values, (batch_size, max_words, max_wordpieces)
        )

    def precompute_batch(
        self,
        start_index: int,
        end_index: int,
        num_neighbors: int,
        random_neighbors: bool = False,
        validation: bool = False,
    ):
        if validation:
            instances = self.validation_instances
        else:
            instances = self.train_instances

        neighbor_indices = self._compute_neighbor_indices(
            start_index,
            end_index,
            num_neighbors=num_neighbors,
            random_neighbors=random_neighbors,
            validation=validation,
        )

        tag_to_neighbors = self._make_tag_to_neighbors_dict(
            neighbor_indices, subsample=True
        )

        max_seq_len = len(instances[start_index])
        max_neighbor_seq_len = max(
            len(self.train_instances[neighbor_index].tags)
            for neighbor_index in neighbor_indices
        )

        targets, target_indices = [], []
        max_correct = 0
        for batch_index, instance in enumerate(instances[start_index:end_index]):
            for word_index in range(max_seq_len):
                true_tag = instance.tags[word_index]
                assert true_tag in tag_to_neighbors

                max_correct = max(max_correct, tag_to_neighbors[true_tag])

                corrects = [
                    sentence_index * max_neighbor_seq_len + token_index
                    for sentence_index, token_index in tag_to_neighbors[true_tag]
                ]
                target_indices.extend(
                    [
                        (batch_index, word_index, correct_index)
                        for correct_index in range(len(corrects))
                    ]
                )
                targets.extend(corrects)

        targets = torch.sparse_coo_tensor(
            list(zip(*target_indices)),
            targets,
            (end_index - start_index, max_seq_len, max_correct),
        )

        x_mapper_sparse = self._get_wordpiece_to_word_mapper(
            list(range(start_index, end_index)), validation=validation
        )

        neighbor_mapper_sparse = self._get_wordpiece_to_word_mapper(neighbor_indices)

        batch = Batch(
            start_index,
            end_index,
            neighbor_indices,
            targets,
            x_mapper_sparse,
            neighbor_mapper_sparse,
        )

        return batch
