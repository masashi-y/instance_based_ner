import argparse
import logging
import random

import hydra
import torch
from omegaconf import OmegaConf
from transformers import (AdamW, AutoModel, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from data import SentDB
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


class TagWordModel(torch.nn.Module):
    def __init__(self, bert_model, dropout):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        for lay in self.bert.encoder.layer:
            lay.output.dropout.p = dropout

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
        bertrep, _ = self.bert(x, attention_mask=mask, output_all_encoded_layers=False)
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
    target_scores = scores.gather(
        1, targets.view(scores.size(0), -1)
    )  # batch_size x max_correct
    loss = -torch.logsumexp(target_scores, dim=1)
    return loss


def get_batch_predictions(batch_representations, neighbor_representations, tag_to_mask):
    """
    batch_representations - batch_size x seq_len x hidden_dim
    neighbor_representations - num_neighbors*neighbor_seq_len x hidden_dim
    tag_to_mask - list of (tag, mask) tuples
    returns batch_size x seq_len tag predictions
    """
    _, seq_len, hidden_size = batch_representations.size()

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


def train(sentdb, model, optimizer, scheduler, device, cfg):
    model.train()
    total_loss, total_preds = 0.0, 0

    for step, batch_index in enumerate(
        torch.randperm(len(sentdb.minibatches)).tolist(), 1
    ):
        optimizer.zero_grad()
        x, neighbors, x_mapper, neigbor_mapper, targets = move_to_device(
            sentdb.train_batch(batch_index), device
        )

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


def do_fscore(sentdb, model, device, cfg):
    """
    micro-avgd segment-level f1-score
    """

    model.eval()
    total_preds, total_golds, total_corrects = 0.0, 0.0, 0.0
    for step in range(len(sentdb.validation_minibatches)):
        x, neighbors, x_mapper, neigbor_mapper, tag_to_mask, golds = move_to_device(
            sentdb.pred_batch(
                step, num_neighbors=cfg.eval_num_neighbors, batch_prediction=True
            ),
            device,
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
            batch_preds, batch_corects = batch_acc_eval(preds, golds)
            batch_golds = batch_preds
        else:
            batch_preds, batch_golds, batch_corects = batch_span_eval(preds, golds)
        total_preds += batch_preds
        total_golds += batch_golds
        total_corrects += batch_corects

    micro_prec = total_corrects / total_preds if total_preds > 0 else 0
    micro_rec = total_corrects / total_golds if total_golds > 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
    return micro_prec, micro_rec, micro_f1


def do_single_fscore(sentdb, model, device, cfg):
    """
    micro-averaged segment-level f1-score
    the point of this is so we don't mix neighbors
    based on other stuff in the minibatch
    """

    model.eval()
    total_preds, total_golds, total_correct = 0.0, 0.0, 0.0
    logger.info("predicting on %d sentences", len(sentdb.validation_instances))
    for step in range(len(sentdb.validation_instances)):

        x, neighbors, x_mapper, neigbor_mapper, tag_to_mask, golds = move_to_device(
            sentdb.pred_batch(
                step, num_neighbors=cfg.eval_num_neighbors, batch_prediction=False
            ),
            device,
        )

        # 1 x seq_len x hidden_dim
        batch_representations = model.get_word_representations(x, x_mapper)
        # num_neighbors x neighbor_seq_len x hidden_dim
        neighbor_representations = model.get_word_representations(
            neighbors, neigbor_mapper, shard_batch_size=cfg.pred_shard_size
        )

        if cfg.cosine:
            neighbor_representations = torch.nn.functional.normalize(
                neighbor_representations, p=2, dim=2
            )
            batch_representations = torch.nn.functional.normalize(
                batch_representations, p=2, dim=2
            )

        preds = get_batch_predictions(
            batch_representations, neighbor_representations, tag_to_mask
        )

        if cfg.eval_accuracy:
            batch_preds, batch_corrects = eval_util.batch_acc_eval(preds, golds)
            batch_golds = batch_preds
        else:
            batch_preds, batch_golds, batch_corrects = eval_util.batch_span_eval(
                preds, golds
            )

        total_preds += batch_preds
        total_golds += batch_golds
        total_correct += batch_corrects

    micro_prec = total_correct / total_preds if total_preds > 0 else 0
    micro_rec = total_correct / total_golds if total_golds > 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
    return micro_prec, micro_rec, micro_f1


@hydra.main(config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))

    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    assert not cfg.zero_shot or cfg.just_eval is not None

    if torch.cuda.is_available() and not cfg.cuda:
        logger.info(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    device = torch.device("cuda" if cfg.cuda else "cpu")
    model = TagWordModel(cfg.bert_model, cfg.dropout).to(device)

    nosplits = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
    if cfg.nosplit_parenth:
        nosplits = nosplits + ("-LPR-", "-RPR-")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.bert_model, do_lower_case=cfg.lower, never_split=nosplits,
    )

    sentdb = SentDB(
        cfg.train_words_file,
        cfg.train_tags_file,
        tokenizer,
        cfg.validation_words_file,
        cfg.validation_tags_file,
        lower=cfg.lower,
        align_strategy=cfg.align_strategy,
        subsample=cfg.subsample,
    )

    nebert = model.bert
    if cfg.zero_shot and "newne" not in cfg.just_eval:
        nebert = AutoModel.from_pretrained(cfg.bert_model).to(device)

    def embedding_fun(x):
        mask = x != 0
        reps, _ = nebert(x, attention_mask=mask.long(), output_all_encoded_layers=False)
        mask = mask.float().unsqueeze(dim=2)
        return (reps * mask).mean(dim=1)

    batch_size, topk = 64, 50  # 128, 500

    # we always compute neighbors w/ cosine; seems to be a bit better
    model.eval()
    sentdb.compute_top_neighbors(
        batch_size,
        embedding_fun,
        topk,
        device,
        cosine=True,
        ignore_train=cfg.zero_shot,
    )

    if not cfg.zero_shot:
        sentdb.make_minibatches(
            cfg.batch_size,
            cfg.neighbor_per_sent,
            random_neighbors_in_train=cfg.random_neighbors_in_train,
        )
        sentdb.make_minibatches(cfg.batch_size, cfg.neighbor_per_sent, validation=True)
        model.train()

    if cfg.just_eval is not None:

        if (
            not cfg.zero_shot and cfg.just_eval != "dev"
        ):  # if zero_shot we already did this
            # need to recompute new neighbors
            if "newne" in cfg.just_eval:
                bert = model.bert
            else:
                bert = AutoModel.from_pretrained(cfg.bert_model).to(device)
            bert.eval()

            # note that ne_bsz and nne are just used for recomputing neighbors
            # and for the max ne stored per sent, resp. we control how many ne
            # are used at prediction time w/ eval_num_neighbors
            sentdb.override_validation_with_test(
                cfg.validation_words_file,
                cfg.validation_tags_file,
                tokenizer,
                embedding_fun,
                device,
                batch_size=128,
                topk=500,
                lower=cfg.lower,
            )

        with torch.no_grad():
            prec, rec, f1 = do_single_fscore(sentdb, model, device, cfg)
            logger.info(
                "Eval: | P: {:3.5f} / R: {:3.5f} / F: {:3.5f}".format(prec, rec, f1)
            )
            exit(0)

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
    num_training_steps = cfg.epochs * len(sentdb.minibatches)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * cfg.warmup_prop),
        num_training_steps=num_training_steps,
    )

    best_f1 = float("inf")
    for epoch in range(cfg.epochs):
        train_loss = train(sentdb, model, optimizer, scheduler, device, cfg)
        logger.info("Epoch {:3d} | train loss {:8.3f}".format(epoch, train_loss))
        with torch.no_grad():
            prec, rec, f1 = do_fscore(sentdb, model, device, cfg)
            logger.info(
                f"Epoch {epoch:3d} | P: {prec:3.5f} / R: {rec:3.5f} / F: {f1:3.5f}"
            )
        if f1 > best_f1:
            best_f1 = f1
            if cfg.save is not None:
                logger.info("saving to %s", cfg.save)
                torch.save({"opt": cfg, "state_dict": model.state_dict()}, cfg.save)


if __name__ == "__main__":
    main()
