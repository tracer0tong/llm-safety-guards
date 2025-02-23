# Code Security Vulnerability Detection

A machine learning-based system for detecting security vulnerabilities in Python code snippets. This project uses a fine-tuned GPT-2 model to classify code into different vulnerability categories, helping developers identify potential security issues in their code.

## Features

- Detects multiple types of security vulnerabilities:
  - SQL Injection
  - Command Injection
  - Path Traversal
  - Cross-Site Scripting (XSS)
  - Safe Code Classification
- Uses pre-trained GPT-2 model fine-tuned on security-focused code examples
- Provides confidence scores for predictions
- Supports both single snippet and batch file analysis
- GPU-accelerated training and inference (when available)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for training)

### Dependencies

Install the required packages using pip:

```bash
pip install torch numpy transformers
```

## Project Structure

```
.
├── main.py                    # Main training and inference code
├── training_samples.json      # Training dataset
├── validation_samples.json    # Validation dataset
└── security_model/           # Directory for saved models
```

## Dataset

The training data consists of JSON files containing code snippets labeled with their security vulnerability categories. Each entry includes:

```json
{
  "category": "VULNERABILITY_TYPE",
  "code": "code_snippet_here"
}
```

Categories include:
- `SQL_INJECTION`
- `COMMAND_INJECTION`
- `PATH_TRAVERSAL`
- `XSS`
- `SAFE`

## Usage

### Training

To train the model on your dataset:

```python
from main import initialize_model, load_training_data, CodeSecurityDataset, train_model

# Initialize model and tokenizer
model, tokenizer = initialize_model()

# Load and prepare training data
train_samples, train_labels = load_training_data('training_samples.json')
train_dataset = CodeSecurityDataset(train_samples, train_labels, tokenizer)

# Train the model
trainer = train_model(model, train_dataset)
```

### Analyzing Code

To analyze code snippets for security vulnerabilities:

```python
from main import CodeSecurityAnalyzer, initialize_model

# Initialize the analyzer
model, tokenizer = initialize_model()
analyzer = CodeSecurityAnalyzer(model, tokenizer)

# Analyze a single code snippet
code_snippet = """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return cursor.execute(query)
"""
result = analyzer.analyze_code(code_snippet)
print(result)  # {'category': 'SQL_INJECTION', 'confidence': 0.95}

# Analyze multiple snippets from a file
results = analyzer.analyze_file('validation_samples.json')
```

## Model Details

The system uses a fine-tuned version of DistilGPT-2 with the following specifications:
- Maximum sequence length: 512 tokens
- Special tokens for code and vulnerability marking
- Classification head for 5 vulnerability categories
- Built on Hugging Face's Transformers library

### Training Parameters

- Learning rate: 5e-5
- Batch size: 8
- Training epochs: 3
- Warmup steps: 100
- Evaluation strategy: Per epoch
- Model selection: Best model based on accuracy

## Training output
```text
{'train_runtime': 11.8075, 'train_samples_per_second': 42.431, 'train_steps_per_second': 5.336, 'train_loss': 0.9835215977260044, 'epoch': 3.0}

Validation Results:

Code:
def delete_user(user_id):
    query = f"DELETE FROM users WHERE id = {user_id}"
    cursor.execute(query)
Predicted: SQL_INJECTION (confidence: 1.00)
Actual: SQL_INJECTION

Code:
def find_files(pattern):
    return os.system(f'find . -name {pattern}')
Predicted: COMMAND_INJECTION (confidence: 0.93)
Actual: COMMAND_INJECTION

Code:
def save_file(filename, content):
    with open(f'uploads/{filename}', 'w') as f:
        f.write(content)
Predicted: PATH_TRAVERSAL (confidence: 0.95)
Actual: PATH_TRAVERSAL

Code:
def display_message(msg):
    return f'<p class="message">{msg}</p>'
Predicted: XSS (confidence: 1.00)
Actual: XSS

Code:
def delete_user(user_id):
    query = "DELETE FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
Predicted: SAFE (confidence: 0.97)
Actual: SAFE

Code:
def save_file(filename, content):
    safe_name = secure_filename(filename)
    path = os.path.join('uploads', safe_name)
    with open(path, 'w') as f:
        f.write(content)
Predicted: SAFE (confidence: 0.63)
Actual: SAFE

Overall Accuracy: 100.00%
```

## Limitations

- The model's accuracy depends on the quality and quantity of training data
- May not detect complex or novel vulnerability patterns
- Limited to Python code analysis
- Maximum sequence length of 512 tokens may truncate longer code snippets

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgments

- Built using Hugging Face's Transformers library
- Uses OpenAI's GPT-2 architecture
- Inspired by various security vulnerability detection systems
