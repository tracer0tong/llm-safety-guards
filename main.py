import json
import numpy as np
import torch
import torch.backends.cuda
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification, TrainingArguments, Trainer

VULN_CATEGORIES = {
    'SQL_INJECTION': 0,
    'COMMAND_INJECTION': 1,
    'PATH_TRAVERSAL': 2,
    'XSS': 3,
    'SAFE': 4
}

class CodeSecurityDataset(Dataset):
    def __init__(self, code_samples, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(code_samples, truncation=True, padding='max_length',
                                   max_length=max_length, return_tensors='pt')
        self.labels = labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach()
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_training_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return ([item['code'] for item in data],
            [VULN_CATEGORIES[item['category']] for item in data])


def initialize_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Disable flash attention
    torch.backends.cuda.enable_flash_sdp(False)

    tokenizer = GPT2TokenizerFast.from_pretrained(
        'distilgpt2',
        model_max_length=512,
        padding_side='right',
        truncation_side='right',
        clean_up_tokenization_spaces=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(
        'distilgpt2',
        num_labels=len(VULN_CATEGORIES),
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True
    )

    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<CODE>', '</CODE>', '<VULNERABLE>', '<SAFE>']
    })
    model.resize_token_embeddings(len(tokenizer))
    return model.to(device), tokenizer


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"accuracy": np.mean(np.argmax(predictions, axis=1) == labels)}


def train_model(model, train_dataset, eval_dataset=None):
    training_args = TrainingArguments(
        output_dir='./security_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        learning_rate=5e-5,
        logging_dir='./logs',
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer


class CodeSecurityAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def analyze_code(self, code_snippet):
        inputs = self.tokenizer(code_snippet, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(predictions[0]).item()

        for category, idx in VULN_CATEGORIES.items():
            if idx == pred_idx:
                return {'category': category, 'confidence': predictions[0][pred_idx].item()}
        return {'category': "UNKNOWN", 'confidence': 0.0}

    def analyze_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        return [{
            'code': item['code'],
            'predicted': analysis['category'],
            'confidence': analysis['confidence'],
            'actual': item.get('category', 'UNKNOWN')
        } for item in data if (analysis := self.analyze_code(item['code']))]


def main():
    model, tokenizer = initialize_model()
    train_samples, train_labels = load_training_data('training_samples.json')
    val_samples, val_labels = load_training_data('validation_samples.json')

    train_dataset = CodeSecurityDataset(train_samples, train_labels, tokenizer)
    val_dataset = CodeSecurityDataset(val_samples, val_labels, tokenizer)

    trainer = train_model(model, train_dataset, val_dataset)
    analyzer = CodeSecurityAnalyzer(model, tokenizer)
    results = analyzer.analyze_file('validation_samples.json')

    correct = sum(1 for r in results if r['predicted'] == r['actual'])
    accuracy = (correct / len(results)) * 100

    print("\nValidation Results:")
    for r in results:
        print(f"\nCode:\n{r['code']}")
        print(f"Predicted: {r['predicted']} (confidence: {r['confidence']:.2f})")
        print(f"Actual: {r['actual']}")
    print(f"\nOverall Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()