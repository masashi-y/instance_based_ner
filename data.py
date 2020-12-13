import dataclasses
import logging
import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Iterator

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__file__)


EmbeddingFun = Callable[[torch.FloatTensor], torch.FloatTensor]


@dataclasses.dataclass
class Alignment:
    start_index: int
    end_index: int


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
    ignore_index: int


@dataclasses.dataclass
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


class SentDB(object):

    align_strategy_choices = ["sum", "first", "last"]

    def __init__(
        self,
        train_sentences_file: str,
        train_tags_file: str,
        tokenizer: PreTrainedTokenizer,
        validation_sentences_file: str,
        validation_tags_file: str,
        lower: bool = False,
        align_strategy: str = "last",
        subsample: int = 2500,
        max_num_neighbors: int = 50,
    ):
        assert (
            align_strategy in self.align_strategy_choices
        ), f"unsupported alignment strategy: {align_strategy}"

        self.align_strategy = align_strategy
        self.subsample = subsample
        self.tokenizer = tokenizer
        self.lower = lower
        self.max_num_neighbors = max_num_neighbors

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

        for batch_instances, _ in _iter_minibatches(instances, batch_size):
            # batch_size x max_wordpieces
            batch = torch.nn.utils.rnn.pad_sequence(
                [torch.LongTensor(instance.wordpieces) for instance in batch_instances],
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

    def get_batch(
        self,
        batch_or_sent_index: int,
        padding_value: int = 0,
        batch_prediction: bool = True,
        validation: bool = False,
    ):
        """
        make a validation minibatch for actually predicting.
        N.B. only uses validation minibatches
        """

        if validation:
            instances = self.validation_instances

            if batch_prediction:
                batch = self.validation_minibatches[batch_or_sent_index]
            else:

                batch = self.precompute_batch(
                    batch_or_sent_index, batch_or_sent_index + 1, validation=True,
                )
        else:
            instances = self.train_instances
            batch = self.train_minibatches[batch_or_sent_index]

            if isinstance(batch, RandomBatch):

                batch = self.precompute_batch(
                    batch.start_index,
                    batch.end_index,
                    random_neighbors=True,
                    validation=False,
                )

        # batch_size x max_wordpieces
        x = torch.nn.utils.rnn.pad_sequence(
            [
                torch.LongTensor(instance.wordpieces)
                for instance in instances[batch.start_index : batch.end_index]
            ],
            padding_value=padding_value,
            batch_first=True,
        )

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

        if not validation:

            sparse_targets = batch.targets
            mask = (
                torch.sparse_coo_tensor(
                    sparse_targets._indices(),
                    torch.ones_like(sparse_targets._values()),
                    sparse_targets.size(),
                )
                .to_dense()
                .bool()
            )
            targets = sparse_targets.to_dense()
            targets[~mask] = batch.ignore_index

            return x, neighbors, x_mapper, neighbor_mapper, targets

        gold_tags = [
            instance.tags
            for instance in self.validation_instances[
                batch.start_index : batch.end_index
            ]
        ]

        tag_to_neighbors = self._make_tag_to_neighbors_dict(
            batch.neighbor_indices,
            subsample=batch_prediction,  # subsample when batch prediction
        )

        num_neighbors = len(batch.neighbor_indices)
        max_neighbor_seq_len = max(
            len(self.train_instances[neighbor_index].tags)
            for neighbor_index in batch.neighbor_indices
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
        self, neighbor_indices: List[int], subsample: bool = True
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        map each tag to location in neighbors
        """

        results = defaultdict(list)
        for sentence_index, neighbor_index in enumerate(neighbor_indices):
            for token_index, tag in enumerate(
                self.train_instances[neighbor_index].tags
            ):
                results[tag].append((sentence_index, token_index))

        if subsample:
            keys = {"O"}
            if self.subsample is not None:
                keys.update(results.keys())

            for key in keys:
                if key in results and len(results[key]) > self.subsample:
                    results[key] = random.sample(results[key], self.subsample)

        return results

    def _compute_neighbor_indices(
        self,
        start_index: int,
        end_index: int,
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
                return random.sample(neighbors, self.max_num_neighbors)
            return neighbors[: self.max_num_neighbors]

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
                neighbor_indices.append(
                    random.choice(
                        [
                            sent_index
                            for sent_index in sent_indices
                            if sent_index not in seen_indices
                        ],
                    )
                )

        return neighbor_indices

    def make_minibatches(
        self,
        batch_size: int,
        random_neighbors_in_train: bool = False,
        validation: bool = False,
    ):
        if validation:
            instances = self.validation_instances
            random_neighbors_in_train = False
        else:
            instances = self.train_instances

        minibatches = []
        for _, (start_index, end_index) in _iter_minibatches(instances, batch_size):
            if random_neighbors_in_train:
                batch = RandomBatch(start_index, end_index)
            else:
                batch = self.precompute_batch(
                    start_index, end_index, validation=validation,
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
        random_neighbors: bool = False,
        validation: bool = False,
    ) -> Batch:
        if validation:
            instances = self.validation_instances
        else:
            instances = self.train_instances

        neighbor_indices = self._compute_neighbor_indices(
            start_index,
            end_index,
            random_neighbors=random_neighbors,
            validation=validation,
        )

        tag_to_neighbors = self._make_tag_to_neighbors_dict(
            neighbor_indices, subsample=True
        )

        batch_size = end_index - start_index
        max_seq_len = len(instances[start_index])
        max_neighbor_seq_len = max(
            len(self.train_instances[neighbor_index].tags)
            for neighbor_index in neighbor_indices
        )

        targets, target_indices = [], []
        max_correct = 0
        for batch_index, instance in enumerate(instances[start_index:end_index]):
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

        x_mapper_sparse = self._get_wordpiece_to_word_mapper(
            list(range(start_index, end_index)), validation=validation
        )

        neighbor_mapper_sparse = self._get_wordpiece_to_word_mapper(neighbor_indices)

        ignore_index = len(neighbor_indices) * max_neighbor_seq_len

        batch = Batch(
            start_index,
            end_index,
            neighbor_indices,
            targets,
            x_mapper_sparse,
            neighbor_mapper_sparse,
            ignore_index,
        )

        return batch
