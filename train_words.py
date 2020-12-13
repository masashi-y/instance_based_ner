import logging
import random

import hydra
import torch
from omegaconf import OmegaConf
from transformers import (AdamW, AutoModel, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from data import CoNLL2003Dataset
from eval_util import accuracy_eval, span_eval

logger = logging.getLogger(__file__)


def move_to_device(obj, cuda_device: torch.device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """

    if cuda_device == torch.device("cpu"):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, cuda_device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, cuda_device) for item in obj)
    else:
        return obj


class Model(torch.nn.Module):
    def __init__(self, bert_model_name: str, dropout: float):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            bert_model_name,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

    def get_word_representations(self, x, mapper, shard_batch_size=None):
        """
        x: batch_size x seq_len
        mapper: batch_size x seq_len x seq_len, binary
        if shard_batch_size is not None, we assume we can detach things
        returns batch_size x seq_len x hidden_dim
        """
        if shard_batch_size is not None:
            return torch.cat(
                [
                    self.get_word_representations(xsplit, mapper_split)
                    for xsplit, mapper_split in zip(
                        torch.split(x, shard_batch_size, dim=0),
                        torch.split(mapper, shard_batch_size, dim=0),
                    )
                ],
                dim=0,
            )

        mask = (x != 0).long()

        # batch_size x seq_len x hidden_dim
        bertrep, _ = self.bert(x, attention_mask=mask)

        # get word reps by selecting or adding pieces
        # batch_size x seq_len x hidden_dim
        results = torch.bmm(mapper, bertrep)
        return results


def get_batch_loss(batch_representations, neighbor_representations, targets):
    """
    batch_representations - batch_size x seq_len x hidden_dim
    neighbor_representations - num_neighbors*neighbor_seq_len x hidden_dim
    targets - batch_size x seq_len x max_correct
    """
    _, _, hidden_size = batch_representations.size()

    # batch_size*seq_len x num_neighbors*neighbor_seq_len
    scores = torch.log_softmax(
        torch.mm(
            batch_representations.view(-1, hidden_size),
            neighbor_representations.view(-1, hidden_size).t(),
        ),
        dim=1,
    )

    dummy = torch.full(scores.size(0), 1, fill_value=float("-inf"))
    scores = torch.cat([scores, dummy], dim=1)
    # batch_size x max_correct
    target_scores = scores.gather(1, targets.view(scores.size(0), -1))
    loss = -torch.logsumexp(target_scores, dim=1)
    return loss


def get_batch_predictions(batch_representations, neighbor_representations, tag_to_mask):
    """
    batch_representations - batch_size x seq_len x hidden_dim
    neighbor_representations - num_neighbors*neighbor_seq_len x hidden_dim
    tag_to_mask - list of (tag, mask) tuples
    returns batch_size x seq_len tag predictions
    """
    batch_size, seq_len, hidden_size = batch_representations.size()

    # batch_size*seq_len x num_neighbors*neighbor_seq_len
    scores = torch.log_softmax(
        torch.mm(
            batch_representations.view(-1, hidden_size),
            neighbor_representations.view(-1, hidden_size).t(),
        ),
        dim=1,
    )
    # sum over all neighbor tokens w/ the same tag
    tag_scores = [torch.logsumexp(scores + mask, dim=1) for tag, mask in tag_to_mask]

    # get a single tag pred for each token
    preds = torch.stack(tag_scores).argmax(dim=0).view(batch_size, seq_len)
    # map back to tags (and transpose)
    index_to_tag = {index: tag for index, (tag, _) in enumerate(tag_to_mask)}
    preds = [[index_to_tag[index.item()] for index in row] for row in preds]

    return preds


def train(dataset, model, optimizer, scheduler, device, cfg):
    model.train()
    total_loss, total_preds = 0.0, 0

    for step, batch in enumerate(dataset.iter_batches(shuffle=True), 1):
        optimizer.zero_grad()
        x, neighbors, x_mapper, neigbor_mapper, targets = move_to_device(batch, device)

        # batch_size x seq_len x hidden_dim
        batch_representations = model.get_word_representations(x, x_mapper)
        if cfg.detach_db:
            with torch.no_grad():
                # num_neighbors x neighbor_seq_len x hidden_dim
                neighbor_representations = model.get_word_representations(
                    neighbors, neigbor_mapper, shard_batch_size=cfg.pred_shard_size
                )
        else:
            neighbor_representations = model.get_word_representations(
                neighbors, neigbor_mapper
            )

        if cfg.cosine:
            batch_representations = torch.nn.functional.normalize(
                batch_representations, p=2, dim=2
            )
            neighbor_representations = torch.nn.functional.normalize(
                neighbor_representations, p=2, dim=2
            )

        loss = get_batch_loss(batch_representations, neighbor_representations, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_preds += x.numel().item()
        if step % cfg.log_interval == 0:
            logger.info("batch %d loss: %f", step, total_loss / total_preds)
    return total_loss / total_preds


def do_fscore(dataset, model, device, cfg, single_prediction=False):
    """
    micro-averaged segment-level f1-score

    single_prediction: the point of this is so we don't mix neighbors
    based on other stuff in the minibatch
    """

    model.eval()
    total_preds, total_golds, total_corrects = 0.0, 0.0, 0.0
    logger.info("predicting on %d sentences", len(dataset.instances))
    for batch in dataset.iter_batches(single_prediction=single_prediction):
        x, neighbors, x_mapper, neigbor_mapper, tag_to_mask, golds = move_to_device(
            batch, device,
        )

        # batch_size x seq_len x hidden_dim
        batch_representations = model.get_word_representations(x, x_mapper)
        # num_neighbors x neighbor_seq_len x hidden_dim
        neighbor_representations = model.get_word_representations(
            neighbors, neigbor_mapper, shard_batch_size=cfg.pred_shard_size
        )
        # batch_size x seq_len
        preds = get_batch_predictions(
            batch_representations, neighbor_representations, tag_to_mask,
        )
        if cfg.eval_accuracy:
            batch_preds, batch_corects = accuracy_eval(preds, golds)
            batch_golds = batch_preds
        else:
            batch_preds, batch_golds, batch_corects = span_eval(preds, golds)
        total_preds += batch_preds
        total_golds += batch_golds
        total_corrects += batch_corects

    micro_prec = total_corrects / total_preds if total_preds > 0 else 0
    micro_rec = total_corrects / total_golds if total_golds > 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
    return micro_prec, micro_rec, micro_f1


@hydra.main(config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))

    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    if torch.cuda.is_available() and cfg.cuda < 0:
        logger.warning(
            "WARNING: You have a CUDA device, so you should probably run with `cuda` option"
        )

    device = torch.device(f"cuda:{cfg.cuda}" if cfg.cuda > -1 else "cpu")

    model = Model(cfg.bert_model, cfg.dropout).to(device)
    if cfg.pretrained is not None:
        logger.info("loading model from %s", cfg.pretrained)
        model.load_state_dict(torch.load(cfg.pretrained, map_location=device))

    nosplits = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
    if cfg.nosplit_parenth:
        nosplits = nosplits + ("-LPR-", "-RPR-")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.bert_model, do_lower_case=cfg.lower, never_split=nosplits,
    )

    def embedding_fun(x):
        mask = x != 0
        reps, _ = model.bert(x, attention_mask=mask.long())
        mask = mask.float().unsqueeze(dim=2)
        return (reps * mask).mean(dim=1)

    model.eval()
    train_dataset = CoNLL2003Dataset(
        "train",
        tokenizer,
        max_num_neighbors=cfg.max_num_neighbors,
        align_strategy=cfg.align_strategy,
        subsample=cfg.subsample,
        random_neighbors_in_train=cfg.random_neighbors_in_train,
    )

    train_dataset.compute_top_neighbors(
        cfg.preprocess_batch_size, embedding_fun, cfg.topk, device, cosine=True,
    )
    train_dataset.make_minibatches(cfg.train_batch_size)

    validation_dataset = CoNLL2003Dataset(
        cfg.validation_data,
        tokenizer,
        max_num_neighbors=cfg.max_num_neighbors,
        align_strategy=cfg.align_strategy,
        subsample=cfg.subsample,
        neighbor_candidate_instances=train_dataset.instances,
    )

    # we always compute neighbors w/ cosine; seems to be a bit better
    validation_dataset.compute_top_neighbors(
        cfg.preprocess_batch_size, embedding_fun, cfg.topk, device, cosine=True,
    )
    validation_dataset.make_minibatches(cfg.train_batch_size)

    if cfg.eval:

        with torch.no_grad():
            prec, rec, f1 = do_fscore(
                validation_dataset, model, device, cfg, single_prediction=True
            )
            logger.info("Eval: | P: %3.5f / R: %3.5f / F: %3.5f", prec, rec, f1)

    else:  # train

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer = AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.01,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=cfg.lr,
            correct_bias=False,
        )
        num_training_steps = cfg.epochs * len(train_dataset.minibatches)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * cfg.warmup_prop),
            num_training_steps=num_training_steps,
        )

        best_f1 = float("inf")
        for epoch in range(cfg.epochs):
            train_loss = train(train_dataset, model, optimizer, scheduler, device, cfg)
            logger.info("Epoch %3d | train loss %8.3f", epoch, train_loss)
            with torch.no_grad():
                prec, rec, f1 = do_fscore(validation_dataset, model, device, cfg)
                logger.info(
                    "Epoch %3d | P: %3.5f / R: %3.5f / F: %3.5f", epoch, prec, rec, f1
                )
            if f1 > best_f1:
                best_f1 = f1
                if cfg.save is not None:
                    logger.info("saving to %s", cfg.save)
                    torch.save(model.state_dict(), cfg.save)


if __name__ == "__main__":
    main()
