import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader
from model import TransformerModel, ModelTrainer
from config import training_config, model_config
from utils.data_loader import create_data_loaders
from utils.logger import setup_logger

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/train_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)
    
    # Initialize model and trainer
    model = TransformerModel().to(device)
    trainer = ModelTrainer(model, device)
    
    # Load data
    train_loader, val_loader = create_data_loaders(
        train_texts=...,
        train_labels=...,
        val_texts=...,
        val_labels=...,
        tokenizer_name=model_config["tokenizer_name"],
        batch_size=training_config["batch_size"],
        max_length=model_config["max_seq_length"]
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(training_config["epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{log_dir}/best_model.pt")
            
        # Early stopping
        if epoch > training_config["patience"] and val_loss > best_val_loss:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
            
    logger.info("Training complete")

if __name__ == "__main__":
    main()
