import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

print("Starting PhoBERT fine-tuning script...")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SentencePairDataset(Dataset):
    """Dataset for sentence pairs and similarity labels"""
    
    def __init__(self, sentence1_list, sentence2_list, labels=None, tokenizer=None, max_length=128):
        self.sentence1_list = sentence1_list
        self.sentence2_list = sentence2_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentence1_list)
    
    def __getitem__(self, idx):
        sentence1 = self.sentence1_list[idx]
        sentence2 = self.sentence2_list[idx]
        
        # Tokenize the sentence pair
        encoded_pair = self.tokenizer(
            sentence1,
            sentence2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to single samples (not batched)
        input_ids = encoded_pair['input_ids'].squeeze()
        attention_mask = encoded_pair['attention_mask'].squeeze()
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

class PhoBERTSimilarityModel(nn.Module):
    """Model for Vietnamese sentence similarity using PhoBERT"""
    
    def __init__(self, phobert_model_name="vinai/phobert-base", num_classes=1):
        super(PhoBERTSimilarityModel, self).__init__()
        self.phobert = AutoModel.from_pretrained(phobert_model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Get the dimension of PhoBERT's hidden layer
        hidden_size = self.phobert.config.hidden_size
        
        # For regression or binary classification 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Sigmoid for binary classification or regression between 0-1
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else None
        
    def forward(self, input_ids, attention_mask):
        # Get the [CLS] token representation which contains the sentence pair info
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Apply sigmoid for binary classification or regression
        if self.sigmoid is not None:
            logits = self.sigmoid(logits)
            
        return logits

def load_data(data_path, test_size=0.2, random_state=42):
    """Load data from CSV file and split into train/validation sets"""
    
    print(f"Reading data from {data_path}")
    try:
        # More robust CSV reading with explicit quoting and error handling
        import csv
        df = pd.read_csv(
            data_path, 
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines='warn', 
            encoding='utf-8', 
            engine='python'
        )
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"First few rows:")
        print(df.head())
        
        # Check if required columns exist
        required_cols = ['sentence1', 'sentence2', 'label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the dataset")
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        return train_df, val_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Let's try a different approach - manually read and parse the CSV
        import csv
        sentences1 = []
        sentences2 = []
        labels = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:  # Make sure we have all three columns
                    sentences1.append(row[0])
                    sentences2.append(row[1])
                    labels.append(float(row[2]))
                else:
                    print(f"Skipping row with insufficient columns: {row}")
        
        # Create a DataFrame manually
        df = pd.DataFrame({
            'sentence1': sentences1,
            'sentence2': sentences2,
            'label': labels
        })
        
        print(f"Data loaded manually. Shape: {df.shape}")
        print(f"First few rows:")
        print(df.head())
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        return train_df, val_df

def train_model(model, train_dataloader, val_dataloader, epochs=10, lr=2e-5, warmup_steps=0):
    """Train the model"""
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Calculate total training steps for the scheduler
    total_steps = len(train_dataloader) * epochs
    
    # Create scheduler with warmup
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function - MSE for regression, CrossEntropy for multi-class
    loss_fn = nn.MSELoss()  # Change to nn.CrossEntropyLoss() for multi-class
    
    # Train loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # Reshape outputs and labels if needed
            if len(outputs.shape) == 2 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
                
            # Calculate loss
            loss = loss_fn(outputs, labels)
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                
                # Reshape outputs and labels if needed
                if len(outputs.shape) == 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                
                # Calculate loss
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    
    return model

def save_model(model, tokenizer, output_dir="./model"):
    """Save the model and tokenizer"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, "phobert_similarity_model_v2.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

def load_saved_model(model_path, tokenizer_path, num_classes=1):
    """Load a saved model and tokenizer"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Initialize model
    model = PhoBERTSimilarityModel(num_classes=num_classes)
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model, tokenizer

def predict_similarity(model, tokenizer, sentence1, sentence2, max_length=128):
    """Predict similarity between two sentences"""
    
    # Prepare input
    encoded_pair = tokenizer(
        sentence1,
        sentence2,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoded_pair['input_ids'].to(device)
    attention_mask = encoded_pair['attention_mask'].to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        # Get the similarity score
        similarity = outputs.item() if outputs.numel() == 1 else outputs.squeeze().tolist()
    
    return similarity

def main():
    """Main function to train and evaluate the model"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fine-tune PhoBERT for Vietnamese sentence similarity')
    parser.add_argument('--data_path', type=str, default="vietnamese_sentence_pairs_example.csv", 
                        help='Path to your CSV file containing sentence pairs')
    parser.add_argument('--model_name', type=str, default="vinai/phobert-base", 
                        help='Pre-trained model name')
    parser.add_argument('--max_length', type=int, default=128, 
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, 
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default="./model", 
                        help='Output directory for saving model and tokenizer')
    
    args = parser.parse_args()
    
    # Parameters
    data_path = args.data_path
    model_name = args.model_name
    max_length = args.max_length
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    output_dir = args.output_dir
    num_classes = 1  # 1 for regression or binary classification
    
    # Load tokenizer
    print("Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and split data
    print(f"Loading and preparing data from {data_path}...")
    train_df, val_df = load_data(data_path)
    
    # Create datasets
    train_dataset = SentencePairDataset(
        train_df['sentence1'].tolist(),
        train_df['sentence2'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length
    )
    
    val_dataset = SentencePairDataset(
        val_df['sentence1'].tolist(),
        val_df['sentence2'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length
    )
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    print(f"Initializing PhoBERT similarity model from {model_name}...")
    model = PhoBERTSimilarityModel(model_name, num_classes=num_classes)
    model.to(device)
    
    # Train model
    print(f"Training model for {epochs} epochs...")
    model = train_model(
        model,
        train_dataloader,
        val_dataloader,
        epochs=epochs,
        lr=learning_rate
    )
    
    # Save model
    print(f"Saving model to {output_dir}...")
    save_model(model, tokenizer, output_dir=output_dir)
    
    # Example prediction
    print("\nExample prediction:")
    example_sentence1 = "Tôi rất thích ăn phở."
    example_sentence2 = "Phở là món ăn yêu thích của tôi."
    similarity = predict_similarity(model, tokenizer, example_sentence1, example_sentence2)
    print(f"Sentence 1: {example_sentence1}")
    print(f"Sentence 2: {example_sentence2}")
    print(f"Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main() 