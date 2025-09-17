import os
import json
import math
import time
import requests
import logging
import torch  # Import torch
import torch.nn as nn  # Import nn from torch
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress TensorFlow warnings if present
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_Engine")

class AdvancedTokenizer:
    """Advanced Byte Pair Encoding Tokenizer"""
    def __init__(self, vocab_size=20000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[SEP]": 4
        }
        self.next_token_id = len(self.special_tokens)
        
    def train(self, texts):
        """Train tokenizer on text corpus"""
        logger.info("Training tokenizer on text corpus...")
        words = Counter()
        for text in texts:
            words.update(text.split())
        
        # Initialize vocabulary with characters
        vocab = Counter()
        for word, count in words.items():
            for char in word:
                vocab[char] += count
        
        # Add most common words as base tokens
        for word, _ in words.most_common(self.vocab_size // 2):
            if word not in vocab:
                vocab[word] = words[word]
        
        # Convert to BPE merges
        self.vocab = {**self.special_tokens}
        for token in vocab:
            if token not in self.vocab and self.next_token_id < self.vocab_size:
                self.vocab[token] = self.next_token_id
                self.next_token_id += 1
        
        # Build inverse vocab
        self.inverse_vocab = {id: token for token, id in self.vocab.items()}
        logger.info(f"Tokenizer trained with {len(self.vocab)} tokens")

    def tokenize(self, text):
        """Tokenize text using greedy BPE-like approach"""
        tokens = []
        words = text.split()
        for word in words:
            # Try to match longest possible token
            start = 0
            while start < len(word):
                end = len(word)
                found = False
                while end > start:
                    substr = word[start:end]
                    if substr in self.vocab:
                        tokens.append(self.vocab[substr])
                        start = end
                        found = True
                        break
                    end -= 1
                if not found:
                    tokens.append(self.vocab["[UNK]"])
                    start += 1
        return tokens

    def decode(self, tokens):
        """Convert token IDs back to text"""
        return " ".join(self.inverse_vocab.get(token, "[UNK]") for token in tokens)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = self.values(values).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)
        
        # Einsum for attention computation
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward"""
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class AdvancedLLM(nn.Module):
    """Advanced Language Model Architecture"""
    def __init__(
        self,
        vocab_size,
        tokenizer,  # Add tokenizer reference
        embed_size=512,
        num_layers=8,
        heads=8,
        forward_expansion=4,
        dropout=0.1,
        max_length=512
    ):
        super().__init__()
        self.tokenizer = tokenizer  # Store tokenizer
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
    
    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        
        out = self.dropout(self.token_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return self.fc_out(out)
    
    def generate(self, context, max_length=100, temperature=0.7, top_k=50):
        """Generate text from context"""
        self.eval()
        tokens = context
        for _ in range(max_length):
            # Truncate to max model length
            input_seq = tokens[-self.max_length:]
            
            # Create causal mask
            seq_len = input_seq.size(0)
            mask = torch.tril(torch.ones(1, seq_len, seq_len)).to(context.device)
            
            # Predict next token
            with torch.no_grad():
                logits = self(input_seq.unsqueeze(0), mask=mask)[0, -1, :]
            
            # Apply temperature and top-k filtering
            logits = logits / temperature
            top_logits, top_indices = logits.topk(top_k)
            probabilities = torch.softmax(top_logits, dim=-1)
            
            # Sample next token
            next_token = top_indices[torch.multinomial(probabilities, 1)]
            tokens = torch.cat([tokens, next_token])
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.vocab["[EOS]"]:
                break
        
        return tokens

class DataEngine:
    """Handles data collection, preprocessing, and augmentation"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.dataset_path = "responses.json"
        self.user_data_path = "user_data.json"
        self.dataset = []
        self.user_data = []
        self.harmful_keywords = [
            "kill", "bomb", "suicide", "attack", "terrorist", 
            "harm", "hurt", "hate", "murder", "abuse"
        ]
        self.load_data()
    
    def load_data(self):
        """Load dataset from files"""
        # Load main dataset
        if os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    self.dataset = json.load(f)
                logger.info(f"Loaded {len(self.dataset)} dataset entries")
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                self.dataset = []
        
        # Load user data
        if os.path.exists(self.user_data_path):
            try:
                with open(self.user_data_path, "r", encoding="utf-8") as f:
                    self.user_data = json.load(f)
                logger.info(f"Loaded {len(self.user_data)} user entries")
            except Exception as e:
                logger.error(f"Error loading user data: {e}")
                self.user_data = []
        else:
            # Create empty user data file if not exists
            try:
                with open(self.user_data_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
                logger.info("Created new user data file")
            except Exception as e:
                logger.error(f"Error creating user data file: {e}")
    
    def contains_harmful_content(self, text):
        """Check for harmful content"""
        lowered = text.lower()
        return any(k in lowered for k in self.harmful_keywords)
    
    def web_search(self, query, max_results=3):
        """Fetch web search results"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            url = f"https://www.bing.com/search?q={requests.utils.quote(query)}"
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            # Extract search results
            for result in soup.select(".b_algo")[:max_results]:
                title = result.select_one("h2")
                snippet = result.select_one(".b_caption p")
                
                if title and snippet:
                    results.append({
                        "title": title.get_text(strip=True),
                        "snippet": snippet.get_text(strip=True)
                    })
            return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    def augment_data(self, query, response):
        """Augment data with web search results"""
        if not response:
            search_results = self.web_search(query)
            if search_results:
                return "\n".join([f"{r['title']}: {r['snippet']}" for r in search_results])
        return response
    
    def prepare_training_data(self, max_length=256):
        """Prepare training data with augmentation"""
        logger.info("Preparing training data...")
        training_texts = []
        
        # Process main dataset
        for entry in self.dataset:
            if "instruction" in entry and "output" in entry:
                text = f"Instruction: {entry['instruction']}\nInput: {entry.get('input', '')}\nOutput: {entry['output']}"
                training_texts.append(text)
        
        # Process user data
        for entry in self.user_data:
            if "query" in entry and "response" in entry:
                augmented = self.augment_data(entry["query"], entry["response"])
                text = f"User: {entry['query']}\nAssistant: {augmented}"
                training_texts.append(text)
        
        # Tokenize and create sequences
        token_sequences = []
        for text in training_texts:
            tokens = self.tokenizer.tokenize(text)
            # Split into chunks of max_length
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i+max_length]
                if len(chunk) < max_length:
                    # Pad with [PAD] tokens
                    pad_id = self.tokenizer.vocab["[PAD]"]
                    chunk += [pad_id] * (max_length - len(chunk))
                token_sequences.append(torch.tensor(chunk, dtype=torch.long))
        
        if token_sequences:
            return torch.stack(token_sequences)
        else:
            logger.warning("No training data available!")
            return torch.tensor([], dtype=torch.long)

class TrainingEngine:
    """Handles model training and evaluation"""
    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["[PAD]"])
    
    def train_epoch(self, data_loader):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0
        for batch in data_loader:
            inputs = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Create shifted output for language modeling
            outputs = self.model(inputs[:, :-1])
            targets = inputs[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        """Evaluate model on validation data"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch.to(self.device)
                outputs = self.model(inputs[:, :-1])
                targets = inputs[:, 1:].contiguous().view(-1)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)

class InferenceEngine:
    """Handles model inference and response generation"""
    def __init__(self, model, tokenizer, data_engine):
        self.model = model
        self.tokenizer = tokenizer
        self.data_engine = data_engine
        self.context_window = []
        self.max_context = 5
    
    def generate_response(self, query):
        """Generate response to user query"""
        # Safety check
        if self.data_engine.contains_harmful_content(query):
            return "\ud83d\udeab I'm here to ensure well-being. Please seek help from appropriate resources."
        
        # Update context
        self.context_window.append(f"User: {query}")
        if len(self.context_window) > self.max_context:
            self.context_window.pop(0)
        
        # Prepare input sequence
        context = "\n".join(self.context_window) + "\nAssistant:"
        input_tokens = self.tokenizer.tokenize(context)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).to(self.model.device)
        
        # Generate response
        output_tokens = self.model.generate(input_tensor, max_length=200, temperature=0.8, top_k=40)
        response = self.tokenizer.decode(output_tokens[len(input_tokens):])
        
        # Clean up response (remove anything after [EOS])
        if "[EOS]" in response:
            response = response.split("[EOS]")[0].strip()
        
        # Add response to context
        self.context_window.append(f"Assistant: {response}")
        
        # Save interaction for future training
        self.save_interaction(query, response)
        
        return response
    
    def save_interaction(self, query, response):
        """Save user interaction for future training"""
        try:
            interaction = {
                "timestamp": time.time(),
                "query": query,
                "response": response
            }
            
            self.data_engine.user_data.append(interaction)
            
            # Save to disk
            with open(self.data_engine.user_data_path, "w", encoding="utf-8") as f:
                json.dump(self.data_engine.user_data, f, ensure_ascii=False, indent=2)
            
            logger.info("Saved user interaction")
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")

# Main training and execution flow
if __name__ == "__main__":
    # Initialize components
    logger.info("Initializing AI engine...")
    
    # Step 1: Initialize tokenizer and train on data
    tokenizer = AdvancedTokenizer(vocab_size=20000)
    data_engine = DataEngine(tokenizer)
    
    # Collect texts for tokenizer training
    texts = []
    for entry in data_engine.dataset:
        if "instruction" in entry and "output" in entry:
            texts.append(f"{entry['instruction']} {entry.get('input', '')} {entry['output']}")
    
    # Also add user data to tokenizer training
    for entry in data_engine.user_data:
        texts.append(entry.get("query", ""))
        texts.append(entry.get("response", ""))
    
    if texts:
        tokenizer.train(texts)
    else:
        logger.warning("No text data available for tokenizer training!")
    
    # Step 2: Prepare training data
    training_data = data_engine.prepare_training_data()
    if training_data.numel() == 0:
        logger.error("No training data available!")
        exit()
    
    # Split into train and validation
    val_size = max(1, int(0.1 * len(training_data)))
    train_data, val_data = training_data[:-val_size], training_data[-val_size:]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)
    
    # Step 3: Initialize and train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AdvancedLLM(
        vocab_size=len(tokenizer.vocab),
        tokenizer=tokenizer,  # Pass tokenizer to model
        embed_size=512,
        num_layers=6,
        heads=8,
        max_length=256
    ).to(device)
    
    trainer = TrainingEngine(model, tokenizer, device)
    
    logger.info("Starting model training...")
    for epoch in range(5):  # Train for 5 epochs
        start_time = time.time()
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1}/5 | Time: {epoch_time:.1f}s | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    logger.info("Training complete!")
    
    # Step 4: Initialize inference engine
    inference_engine = InferenceEngine(model, tokenizer, data_engine)
    
    # Interactive session
    print("Advanced LLM Assistant initialized. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nAssistant: Goodbye! Have a great day.")
                break
            
            response = inference_engine.generate_response(user_input)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nSession ended.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.exception("Error during inference")