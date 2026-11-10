"""Bai 2: fine-tune DistilBERT tren IMDb, validation rieng va test giu lai."""
import argparse
import json
from pathlib import Path
from metrics_utils import binary_metrics

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-size', type=int, default=2000)
    p.add_argument('--val-size', type=int, default=500)
    p.add_argument('--test-size', type=int, default=500)
    p.add_argument('--epochs', type=float, default=2)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='outputs/finetune')
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    if min(a.train_size, a.val_size, a.test_size, a.batch_size) < 1 or a.epochs <= 0:
        p.error('Sizes, batch size and epochs must be positive.')
    if not 8 <= a.max_length <= 512:
        p.error('max-length must be between 8 and 512.')
    return a

def main(a):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datasets import load_dataset, DatasetDict
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)
    set_seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_dataset('stanfordnlp/imdb')
    # Chia validation tu TRAIN; tap TEST chi dung danh gia cuoi cung.
    if a.train_size + a.val_size > len(raw['train']) or a.test_size > len(raw['test']):
        raise ValueError('IMDb allows train-size + val-size <= 25000 and test-size <= 25000.')
    pool = raw['train'].shuffle(seed=a.seed)
    data = DatasetDict({
        'train': pool.select(range(a.train_size)),
        'validation': pool.select(range(a.train_size, a.train_size + a.val_size)),
        'test': raw['test'].shuffle(seed=a.seed).select(range(a.test_size))})
    print('Device:', 'cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    print(data)
    name = 'distilbert/distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(name)
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=a.max_length)
    encoded = data.map(tokenize, batched=True, remove_columns=['text'])
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2, id2label={0: 'NEGATIVE', 1: 'POSITIVE'},
        label2id={'NEGATIVE': 0, 'POSITIVE': 1})
    def compute_metrics(result):
        return binary_metrics(result.label_ids.tolist(), np.argmax(result.predictions, axis=-1).tolist())
    training_args = TrainingArguments(
        output_dir=str(out / 'checkpoints'), learning_rate=2e-5,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        num_train_epochs=a.epochs, weight_decay=0.01,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model='f1', greater_is_better=True,
        logging_steps=10, logging_first_step=True, report_to='none',
        seed=a.seed, data_seed=a.seed, use_cpu=a.cpu,
        fp16=False, dataloader_num_workers=0, push_to_hub=False)
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=encoded['train'], eval_dataset=encoded['validation'],
                      processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=compute_metrics)
    trainer.train()
    # load_best_model_at_end da nap checkpoint tot nhat theo validation F1.
    result = trainer.predict(encoded['test'], metric_key_prefix='test')
    pred = result.predictions.argmax(axis=-1)
    report = classification_report(result.label_ids, pred, labels=[0, 1],
                                   target_names=['NEGATIVE', 'POSITIVE'], zero_division=0)
    print(report)
    print(result.metrics)
    trainer.save_model(str(out / 'model'))
    tokenizer.save_pretrained(out / 'model')
    for filename, value in [('metrics.json', result.metrics), ('config.json', vars(a)),
                            ('training_history.json', trainer.state.log_history)]:
        (out / filename).write_text(json.dumps(value, indent=2), encoding='utf-8')
    (out / 'classification_report.txt').write_text(report, encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(result.label_ids, pred, labels=[0, 1],
        display_labels=['NEGATIVE', 'POSITIVE'], cmap='Blues')
    plt.title('IMDb — held-out test set')
    plt.tight_layout()
    plt.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close()
    fig, ax = plt.subplots()
    for key, label in [('loss', 'Train loss'), ('eval_loss', 'Validation loss')]:
        rows = [x for x in trainer.state.log_history if key in x]
        ax.plot([x['epoch'] for x in rows], [x[key] for x in rows], marker='o', label=label)
    ax.set(xlabel='Epoch', ylabel='Loss', title='Training progress')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'loss_curve.png', dpi=160)
    plt.close(fig)
    print('Saved results to:', out.resolve())

if __name__ == '__main__':
    main(parse_args())
