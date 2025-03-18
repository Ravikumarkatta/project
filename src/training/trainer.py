# src/training/trainer.py
import torch
from torch.utils.data import DataLoader
import json
import os
import sys
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

# Use absolute imports
from src.model.architecture import BiblicalTransformer, BiblicalTransformerConfig
from src.data.preprocessing import load_processed_data
from src.training.loss import TheologicalLoss
from src.training.optimization import get_optimizer_and_scheduler
from src.utils.logger import setup_logger
from src.monitoring.metrics import MetricsCollector
from src.theology.validator import TheologicalValidator

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check for __init__.py files
init_files_present = True
for subdir in ['src', 'src/model', 'src/data', 'src/training', 'src/utils']:
    init_file = os.path.join(PROJECT_ROOT, subdir, '__init__.py')
    if not os.path.exists(init_file):
        print(f"Error: Missing __init__.py in {subdir}")
        init_files_present = False
        break

if not init_files_present:
    print("Error: Missing __init__.py files. Please create them.")
    sys.exit(1)

class Trainer:
    def __init__(self, config_path: str):
        # Load configuration with validation
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self._validate_config()

        # Initialize early stopping counter
        self.early_stopping_counter = 0
        
        # Create model config
        self.model_config = BiblicalTransformerConfig(**self.config.get('model_params', {}))
        
        # Setup device and logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log_dir = self.config.get('logging', {}).get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger("trainer", os.path.join(log_dir, "training.log"))
        
        # Initialize components
        self.setup_model()
        self.setup_data()
        self.setup_training_components()
        self.metrics_collector = MetricsCollector(port=8000, validator_config={
            "min_score": 0.9,
            "theological_terms": ["God", "Jesus", "Holy Spirit"]
        })
        self.validator = TheologicalValidator(self.metrics_collector.validator_config)

    def _validate_config(self):
        """Validate the config file structure."""
        required_keys = ['model_params', 'data', 'training', 'optimizer', 'logging']
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise KeyError(f"Missing required config keys: {missing_keys}")

    def setup_model(self):
        """Initialize model with error handling."""
        try:
            self.model = BiblicalTransformer(self.model_config).to(self.device)
            self.logger.info(f"Model initialized successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def setup_data(self):
        """Initialize data loaders with proper path handling."""
        try:
            data_path = self.config.get('data', {}).get('data_path', 'data/processed')
            self.train_data, self.val_data = load_processed_data(
                os.path.join(PROJECT_ROOT, data_path)
            )
            batch_size = self.config['training']['batch_size']
            self.train_loader = DataLoader(
                self.train_data, 
                batch_size=batch_size,
                shuffle=True
            )
            self.val_loader = DataLoader(
                self.val_data,
                batch_size=batch_size,
                shuffle=False
            )
            self.logger.info(f"Loaded {len(self.train_data)} training samples and {len(self.val_data)} validation samples")
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def setup_training_components(self):
        """Initialize loss, optimizer, and scheduler with config params."""
        try:
            loss_config = self.config.get('loss', {})
            main_loss_type = loss_config.get('main_loss', 'cross_entropy')
            self.logger.info(f"Using {main_loss_type} as main loss function")
            self.criterion = TheologicalLoss()
            
            lr = self.config['optimizer']['learning_rate']
            warmup_steps = self.config['training']['warmup']['warmup_steps']
            epochs = self.config['training']['max_epochs']
            total_steps = len(self.train_loader) * epochs
            
            self.optimizer, self.scheduler = get_optimizer_and_scheduler(
                self.model.parameters(),
                lr=lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps
            )
            self.logger.info(f"Initialized optimizer with learning rate {lr} and {warmup_steps} warmup steps")
        except KeyError as e:
            self.logger.error(f"Missing configuration key: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize training components: {str(e)}")
            raise

    def train(self):
        """Training loop with theological validation and monitoring."""
        max_epochs = self.config['training']['max_epochs']
        best_val_loss = float('inf')
        max_grad_norm = self.config['training']['max_grad_norm']
        accumulation_steps = self.config['training']['accumulation_steps']
        use_mixed_precision = self.config['training']['mixed_precision']
        scaler = GradScaler() if use_mixed_precision else None
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{max_epochs}")
            self.model.train()
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids, labels, attention_mask, verse_positions = batch  # Unpack verse_positions
                
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                verse_positions = verse_positions.to(self.device)
                
                if batch_idx % accumulation_steps == 0:
                    self.optimizer.zero_grad()
                
                if use_mixed_precision:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            verse_positions=verse_positions,
                            labels=labels
                        )
                        loss = self.criterion(outputs['logits'], labels) / accumulation_steps
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.scheduler.step()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        verse_positions=verse_positions,
                        labels=labels
                    )
                    loss = self.criterion(outputs['logits'], labels) / accumulation_steps
                    loss.backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                
                epoch_loss += loss.item() * accumulation_steps
                
                log_every_n_steps = self.config['logging']['log_every_n_steps']
                if batch_idx % log_every_n_steps == 0:
                    self.logger.info(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}")
                
                # Track metrics
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                with torch.no_grad():
                    predicted_ids = outputs['logits'].argmax(dim=-1)
                    predicted_text = self.model.tokenizer.detokenize(predicted_ids)  # Assume tokenizer is accessible
                    validation_score = self.validator.validate({"text": predicted_text})
                end_time.record()
                torch.cuda.synchronize()
                latency = start_time.elapsed_time(end_time) / 1000.0
                self.metrics_collector.track_inference(latency)
                self.metrics_collector.track_validation_score({"validation_score": validation_score})
                self.metrics_collector.track_verse_accuracy(outputs['verse_logits'], verse_positions)
                
                save_every_n_steps = self.config['training']['checkpoint']['save_every_n_steps']
                if batch_idx > 0 and batch_idx % save_every_n_steps == 0:
                    checkpoint_dir = self.config.get('model', {}).get('checkpoint_dir', 'models/checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}_step{batch_idx}.pt")
                    torch.save({
                        'epoch': epoch,
                        'step': batch_idx,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, checkpoint_path)
                    self.logger.info(f"Saved checkpoint at epoch {epoch+1}, step {batch_idx}")
            
            avg_train_loss = epoch_loss / len(self.train_loader)
            val_loss = self.validate()
            self.logger.info(f"Epoch {epoch+1}/{max_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            save_best_only = self.config['training']['checkpoint']['save_best_only']
            if save_best_only and val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = self.config.get('model', {}).get('save_path', 'models/best_model.pt')
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_save_path)
                self.logger.info(f"Saved best model with Val Loss: {best_val_loss:.4f}")
            
            if self.config['training']['early_stopping']['enabled']:
                patience = self.config['training']['early_stopping']['patience']
                min_delta = self.config['training']['early_stopping']['min_delta']
                if val_loss > best_val_loss - min_delta:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                else:
                    self.early_stopping_counter = 0

    def validate(self):
        """Validation loop with consistent output handling."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, labels, attention_mask, verse_positions = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                verse_positions = verse_positions.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    verse_positions=verse_positions,
                    labels=labels
                )
                loss = self.criterion(outputs['logits'], labels)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a BiblicalTransformer model")
    parser.add_argument("--config", type=str, default="config/training_config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()