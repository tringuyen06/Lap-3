"""Chi so nhi phan, thu tu nhan 0=NEGATIVE, 1=POSITIVE."""
def binary_metrics(labels, predictions):
    labels, predictions = list(labels), list(predictions)
    if not labels or len(labels) != len(predictions):
        raise ValueError('Labels and predictions must have equal, nonzero length.')
    if any(x not in (0, 1) for x in labels + predictions):
        raise ValueError('Only binary labels 0 and 1 are supported.')
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'accuracy': sum(y == p for y, p in zip(labels, predictions)) / len(labels),
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
