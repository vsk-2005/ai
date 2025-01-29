import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os

class TextGenerator:
    def __init__(self, hidden_size=256):
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def _build_model(self, vocab_size):
        class RNNModel(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super(RNNModel, self).__init__()
                self.hidden_size = hidden_size
                self.vocab_size = vocab_size
                
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, x, hidden=None):
                batch_size = x.size(0)
                
                if hidden is None:
                    hidden = self.init_hidden(batch_size)
                
                embedded = self.embedding(x)
                output, hidden = self.lstm(embedded, hidden)
                output = self.fc(output)
                return output, hidden
            
            def init_hidden(self, batch_size):
                weight = next(self.parameters())
                return (weight.new_zeros(1, batch_size, self.hidden_size),
                       weight.new_zeros(1, batch_size, self.hidden_size))
        
        self.model = RNNModel(vocab_size, self.hidden_size).to(self.device)
        return self.model

    def prepare_data(self, text):
        # Create vocabulary
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Convert text to indices
        indices = [self.char_to_idx[ch] for ch in text]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def create_batches(self, data, sequence_length, batch_size):
        """Create batches from data"""
        # Calculate total number of sequences
        num_sequences = len(data) - sequence_length
        if num_sequences <= 0:
            raise ValueError(f"Text length ({len(data)}) must be greater than sequence_length ({sequence_length})")
            
        # Create sequences and targets
        sequences = []
        targets = []
        
        # Step through data to create sequences
        for i in range(0, num_sequences, 1):  # Changed step size to 1 for more sequences
            seq = data[i:i + sequence_length]
            target = data[i + 1:i + sequence_length + 1]
            sequences.append(seq)
            targets.append(target)
        
        # Convert to tensors
        sequences = torch.stack(sequences).to(self.device)
        targets = torch.stack(targets).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(sequences, targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # Ensure all batches are full size
        )
        
        return dataloader

    def train(self, text, sequence_length=100, batch_size=32, num_epochs=100, learning_rate=0.001):
        """Train the model on the given text"""
        if not text:
            raise ValueError("Training text cannot be empty")
            
        print(f"Training on {self.device}")
        print(f"Text length: {len(text)} characters")
        
        # Prepare data
        data = self.prepare_data(text)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Verify we have enough data
        if len(data) <= sequence_length:
            raise ValueError(f"Text length ({len(data)}) must be greater than sequence_length ({sequence_length})")
        
        # Build model if not exists
        if self.model is None:
            self._build_model(self.vocab_size)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        try:
            # Create dataloader
            dataloader = self.create_batches(data, sequence_length, batch_size)
            
            if len(dataloader) == 0:
                raise ValueError("No batches created. Try reducing batch_size or sequence_length")
            
            print(f"Created {len(dataloader)} batches")
            
            for epoch in range(num_epochs):
                total_loss = 0
                batch_count = 0
                
                for batch_sequences, batch_targets in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output, _ = self.model(batch_sequences)
                    
                    # Compute loss
                    loss = criterion(output.view(-1, self.vocab_size), batch_targets.view(-1))
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:  # Add check to prevent division by zero
                    avg_loss = total_loss / batch_count
                    if (epoch + 1) % 10 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
                else:
                    print("Warning: No batches processed in this epoch")
                    
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            raise

    def generate(self, prompt, max_length=200, temperature=0.8):
        if not self.model:
            raise ValueError("Model not trained yet")
            
        self.model.eval()
        with torch.no_grad():
            # Convert prompt to tensor
            current = torch.tensor([self.char_to_idx[ch] for ch in prompt], 
                                dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generate characters
            generated_text = prompt
            hidden = None
            
            for _ in range(max_length):
                output, hidden = self.model(current, hidden)
                
                # Apply temperature
                output = output[:, -1, :] / temperature
                probs = torch.softmax(output, dim=-1)
                
                # Sample from the distribution
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # Append to generated text
                generated_text += self.idx_to_char[next_char_idx]
                
                # Update current
                current = torch.tensor([[next_char_idx]], dtype=torch.long, device=self.device)
                
            return generated_text

    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state and vocabulary
        torch.save({
            'model_state': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size
        }, path)

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
            
        # Load the saved state
        checkpoint = torch.load(path)
        
        # Restore vocabulary
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.vocab_size = checkpoint['vocab_size']
        self.hidden_size = checkpoint['hidden_size']
        
        # Rebuild model and load state
        self._build_model(self.vocab_size)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing text generator...")
    generator = TextGenerator(hidden_size=256)
    
    # Get the current directory and construct absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_file = os.path.join(current_dir, 'wiki_01.txt')
    model_file = os.path.join(current_dir, 'model.pth')
    
    # Read training data
    try:
        print(f"Reading training data from {training_file}")
        with open(training_file, 'r', encoding='utf-8') as f:
            training_text = f.read()
        
        if not training_text:
            raise ValueError("Training file is empty")
            
        print("Starting training...")
        generator.train(
            text=training_text,
            sequence_length=100,
            batch_size=32,
            num_epochs=100
        )
        
        # Save the trained model
        generator.save_model(model_file)
        print("Training completed and model saved!")
        
        # Generate sample text
        sample = generator.generate("The future", max_length=200)
        print("\nSample generated text:")
        print(sample)
        
    except FileNotFoundError:
        print(f"Error: Could not find training data file at {training_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")