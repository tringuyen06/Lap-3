"""Du doan voi mo hinh da fine-tune va luu tren may."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='outputs/finetune/model')
    p.add_argument('--text', nargs='+', required=True)
    a = p.parse_args()
    if not (Path(a.model) / 'config.json').is_file():
        p.error('Model not found. Run exercise2_finetune.py first or pass --model.')
    import torch
    from transformers import pipeline
    classifier = pipeline('text-classification', model=a.model, tokenizer=a.model,
                          device=0 if torch.cuda.is_available() else -1)
    for text, result in zip(a.text, classifier(a.text, truncation=True, max_length=128)):
        print(f"{text} -> {result['label']} ({result['score']:.4f})")

if __name__ == '__main__':
    main()
