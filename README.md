# Code Security Vulnerability Detection

A demo of machine learning-based system for detecting security vulnerabilities in Python code snippets. This project uses a fine-tuned GPT-2 model to classify code into different vulnerability categories, helping developers identify potential security issues in their code.

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
