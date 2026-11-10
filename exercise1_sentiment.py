"""Bai 1: tokenize va phan tich cam xuc bang mo hinh da huan luyen."""
import argparse
import json
from pathlib import Path

def run(texts, output='outputs/exercise1.json'):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    with torch.inference_mode():
        probs = model(**{k: v.to(device) for k, v in inputs.items()}).logits.softmax(dim=-1).cpu()
    rows = []
    for text, prob in zip(texts, probs):
        label = int(prob.argmax())
        rows.append({'text': text, 'tokens': tokenizer.tokenize(text),
                     'input_ids': tokenizer(text, truncation=True, max_length=256)['input_ids'],
                     'label': model.config.id2label[label], 'score': float(prob[label])})
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', nargs='+', default=['I really enjoyed this movie!', 'The movie was boring and disappointing.'])
    args = parser.parse_args()
    run(args.text)
