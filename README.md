# PhoBERT Vietnamese Sentence Similarity

This project fine-tunes the PhoBERT model for evaluating semantic similarity between Vietnamese sentences.

## Requirements

```
torch>=1.7.0
transformers>=4.5.0
pandas
numpy
scikit-learn
tqdm
```

Install dependencies:

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## Dataset Format

The model requires a CSV file with the following columns:

- `sentence1`: First Vietnamese sentence
- `sentence2`: Second Vietnamese sentence
- `label`: Similarity score (e.g., values between 0 and 1 where 1 is most similar)

Example: `vietnamese_sentence_pairs_example.csv`

## Usage

### Training the Model

```python
python phobert_sentence_similarity.py
```

By default, the script will:

1. Load data from `vietnamese_sentence_similarity.csv`
2. Split it into training and validation sets
3. Fine-tune PhoBERT for 3 epochs
4. Save the model to `./model/`

### Using a Trained Model

```python
import torch
from phobert_sentence_similarity import load_saved_model, predict_similarity

# Load model and tokenizer
model_path = "./model/phobert_similarity_model.pth"
tokenizer_path = "./model/tokenizer"
model, tokenizer = load_saved_model(model_path, tokenizer_path)

# Predict similarity between two sentences
sentence1 = "Tôi rất thích ăn phở."
sentence2 = "Phở là món ăn yêu thích của tôi."
similarity = predict_similarity(model, tokenizer, sentence1, sentence2)
print(f"Similarity: {similarity:.4f}")
```

## Customization

You can modify the following parameters in the `main()` function:

- `data_path`: Path to your CSV dataset
- `max_length`: Maximum sequence length (default: 128 tokens)
- `batch_size`: Batch size for training (default: 16)
- `epochs`: Number of training epochs (default: 3)
- `learning_rate`: Learning rate (default: 2e-5)
- `num_classes`: 1 for regression/binary classification, >1 for multi-class classification

For multi-class classification, modify the loss function in `train_model()` from `nn.MSELoss()` to `nn.CrossEntropyLoss()`.

## Model Architecture

The model uses the pretrained `vinai/phobert-base` as the base and adds:

- A dropout layer (0.1)
- A classifier head with:
  - Linear layer (768 -> 256)
  - ReLU activation
  - Dropout (0.1)
  - Linear layer (256 -> num_classes)
  - Sigmoid activation (for regression/binary classification)

## Acknowledgements

- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) by VinAI Research
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
