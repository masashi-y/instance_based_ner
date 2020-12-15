from typing import List, Set, Tuple


Span = Tuple[int, int, str]


# much of this adapted from https://github.com/iesl/dilated-cnn-ner/blob/08abe11aa8ecfd6eb499b959b5386f2d8ca0602e/src/eval_f1.py
def get_spans(tags: List[str]) -> Set[Span]:
    def is_start(curr):
        return curr[0] == "B" or curr[0] == "U"

    def is_continue(curr):
        return curr[0] == "I" or curr[0] == "L"

    def is_background(curr):
        return not is_start(curr) and not is_continue(curr)

    def is_seg_start(curr, prev):
        return is_start(curr) or (
            is_continue(curr)
            and (prev is None or is_background(prev) or prev[1:] != curr[1:])
        )

    results = set()
    in_span, span_type = None, None
    tags = [None] + tags

    for index, (curr, prev) in enumerate(zip(tags[1:], tags[:-1])):

        if is_seg_start(curr, prev):
            if in_span is not None:
                results.add((in_span, index, span_type))
            in_span, span_type = index, curr[2:]
        elif not is_continue(curr):
            if in_span is not None:
                results.add((in_span, index, span_type))
            in_span, span_type = None, None

    if in_span is not None:
        results.add((in_span, len(tags) - 1, span_type))
    return results


def span_eval(preds: List[List[str]], golds: List[List[str]]) -> Tuple[int, int, int]:
    """
    both preds and golds are bsz x T (but are lists)
    """

    assert len(golds) == len(preds)
    num_preds, num_golds, num_corrects = 0, 0, 0

    for pred, gold in zip(preds, golds):
        pred_spans = get_spans(pred)
        gold_spans = get_spans(gold)
        num_preds += len(pred_spans)
        num_golds += len(gold_spans)
        num_corrects += len(pred_spans & gold_spans)

    return num_preds, num_golds, num_corrects


def accuracy_eval(preds: List[List[str]], golds: List[List[str]]) -> Tuple[int, int]:
    assert len(golds) == len(preds)
    correct = sum(
        pred_tag == gold_tag
        for pred, gold in zip(preds, golds)
        for pred_tag, gold_tag in zip(pred, gold)
    )
    return len(preds) * len(preds[0]), correct
