# src/training/trainer.py
import torch
from torch.utils.data import DataLoader
import json
import os
import sys
from typing import Tuple, Dict
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

# Absolute imports
from src.model.architecture import BiblicalTransformer, BiblicalTransformerConfig
from src.data.preprocessing import load_processed_data
from src.data.dataset import BibleDataset  # Assuming a custom dataset class
from src.training.loss import TheologicalLoss
from src.training.optimization import get_optimizer_and_scheduler
from src.utils.logger import setup_logger
from src.theology.validator import TheologicalValidator  # For theological checks
from src.quality.theological_eval import evaluate_theological_accuracy

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check __init__.py files
for subdir in ['src', 'src/model', 'src/data', 'src/training', 'src/utils', 'src/theology', 'src/quality']:
    if not os.path.exists(os.path.join(PROJECT_ROOT, subdir, '__init__.py')):
        print(f"Error: Missing __init__.py in {subdir}")
        sys.exit(1)

class Trainer:
    def __init__(self, config_path: str, checkpoint_path: str = None):
        """Initialize trainer with config and optional checkpoint."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_dir = self.config.get('logging', {}).get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger("trainer", os.path.join(log_dir, "training.log"))
        self.writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))  # TensorBoard
        
        self.model_config = BiblicalTransformerConfig(**self.config.get('model_params', {}))
        self.theological_validator = TheologicalValidator(self.config.get('theology', {}))
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        self.setup_model()
        self.setup_data()
        self.setup_training_components()
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def setup_model(self):
        """Initialize model."""
        try:
            self.model = BiblicalTransformer(self.model_config).to(self.device)
            self.logger.info(f"Model initialized on {self.device}")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise

    def setup_data(self):
        """Initialize data loaders with multi-translation support."""
        try:
            data_config = self.config.get('data', {})
            data_path = os.path.join(PROJECT_ROOT, data_config.get('data_path', 'data/processed'))
            train_data, val_data = load_processed_data(data_path)
            batch_size = self.config.get('training', {}).get('batch_size', 16)
            
            self.train_loader = DataLoader(
                BibleDataset(train_data, data_config),  # Custom dataset for flexibility
                batch_size=batch_size,
                shuffle=True,
                num_workers=data_config.get('num_workers', 0)
            )
            self.val_loader = DataLoader(
                BibleDataset(val_data, data_config),
                batch_size=batch_size,
                shuffle=False,
                num_workers=data_config.get('num_workers', 0)
            )
            self.logger.info(f"Loaded {len(self.train_loader.dataset)} train, {len(self.val_loader.dataset)} val samples")
        except Exception as e:
            self.logger.error(f"Data setup failed: {e}")
            raise

    def setup_training_components(self):
        """Initialize training components."""
        try:
            loss_config = self.config.get('loss', {})
            self.criterion = TheologicalLoss(**loss_config)
            self.logger.info(f"Using TheologicalLoss with config: {loss_config}")
            
            optim_config = self.config.get('optimizer', {})
            lr = optim_config.get('learning_rate', 2e-5)
            warmup_steps = self.config.get('training', {}).get('warmup', {}).get('warmup_steps', 500)
            total_steps = len(self.train_loader) * self.config.get('training', {}).get('max_epochs', 10)
            
            self.optimizer, self.scheduler = get_optimizer_and_scheduler(
                self.model.parameters(), lr=lr, warmup_steps=warmup_steps, total_steps=total_steps
            )
            self.scaler = torch.cuda.amp.GradScaler() if self.config.get('training', {}).get('mixed_precision', False) else None
            self.logger.info(f"Optimizer: lr={lr}, warmup_steps={warmup_steps}")
        except Exception as e:
            self.logger.error(f"Training components setup failed: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str):
        """Load model state from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {checkpoint['epoch']}")
        except Exception as e:
            self.logger.error(f"Checkpoint loading failed: {e}")
            raise

    def train(self) -> None:
        """Training loop with theological validation and logging."""
        training_config = self.config.get('training', {})
        max_epochs = training_config.get('max_epochs', 10)
        max_grad_norm = training_config.get('max_grad_norm', 1.0)
        accumulation_steps = training_config.get('accumulation_steps', 1)
        use_mixed_precision = training_config.get('mixed_precision', False)
        log_interval = self.config.get('logging', {}).get('log_every_n_steps', 10)
        
        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0
            theological_scores = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids, labels, attention_mask = batch['input_ids'].to(self.device), batch['labels'].to(self.device), batch['attention_mask'].to(self.device)
                
                if batch_idx % accumulation_steps == 0:
                    self.optimizer.zero_grad()
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = self.criterion(outputs['logits'], labels) / accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs['logits'], labels) / accumulation_steps
                    loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    if use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.optimizer.step()
                    self.scheduler.step()
                
                epoch_loss += loss.item() * accumulation_steps
                if batch_idx % log_interval == 0:
                    self.logger.info(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}")
                    self.writer.add_scalar("Train/Loss", loss.item() * accumulation_steps, epoch * len(self.train_loader) + batch_idx)
                
                # Theological validation (sample every 50 steps)
                if batch_idx % 50 == 0:
                    theo_score = self.theological_validator.validate(outputs['logits'], labels)
                    theological_scores.append(theo_score)
                    self.writer.add_scalar("Train/TheologicalScore", theo_score, epoch * len(self.train_loader) + batch_idx)
                
                # Checkpointing
                if batch_idx > 0 and batch_idx % training_config.get('checkpoint', {}).get('save_every_n_steps', 1000) == 0:
                    self.save_checkpoint(epoch, batch_idx)

            avg_train_loss = epoch_loss / len(self.train_loader)
            avg_theo_score = sum(theological_scores) / len(theological_scores) if theological_scores else 0
            self.logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Theo Score = {avg_theo_score:.4f}")
            self.writer.add_scalar("Train/EpochLoss", avg_train_loss, epoch)
            
            val_loss, val_theo_score = self.validate()
            self.logger.info(f"Validation: Loss = {val_loss:.4f}, Theo Score = {val_theo_score:.4f}")
            self.writer.add_scalar("Validation/Loss", val_loss, epoch)
            self.writer.add_scalar("Validation/TheologicalScore", val_theo_score, epoch)
            
            if training_config.get('checkpoint', {}).get('save_best_only', False) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_save_path = self.config.get('model', {}).get('save_path', 'models/best_model.pt')
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_save_path)
                self.logger.info(f"Saved best model with Val Loss: {val_loss:.4f}")
            
            if training_config.get('early_stopping', {}).get('enabled', False):
                patience = training_config.get('early_stopping', {}).get('patience', 3)
                min_delta = training_config.get('early_stopping', {}).get('min_delta', 0.001)
                if val_loss > self.best_val_loss - min_delta:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                else:
                    self.early_stopping_counter = 0

        self.writer.close()

    def validate(self) -> Tuple[float, float]:
        """Validation loop with theological evaluation."""
        self.model.eval()
        val_loss = 0.0
        theological_scores = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, labels, attention_mask = batch['input_ids'].to(self.device), batch['labels'].to(self.device), batch['attention_mask'].to(self.device)
                if self.config.get('training', {}).get('mixed_precision', False):
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = self.criterion(outputs['logits'], labels)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs['logits'], labels)
                val_loss += loss.item()
                theo_score = self.theological_validator.validate(outputs['logits'], labels)
                theological_scores.append(theo_score)
        
        avg_val_loss = val_loss / len(self.val_loader)
        avg_theo_score = sum(theological_scores) / len(theological_scores) if theological_scores else 0
        return avg_val_loss, avg_theo_score

    def save_checkpoint(self, epoch: int, step: int):
        """Save training checkpoint."""
        checkpoint_dir = self.config.get('model', {}).get('checkpoint_dir', 'models/checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}_step{step}.pt")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint at {checkpoint_path}")
    def validate(self):
    results = evaluate_model(
        self.model, self.val_loader, self.config, self.device, epoch=epoch)
    return results["loss"], results["theological_alignment"]
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a BiblicalTransformer model")
    parser.add_argument("--config", type=str, default="config/training_config.json", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume training")
    args = parser.parse_args()
    
    trainer = Trainer(args.config, args.checkpoint)
    trainer.train()
