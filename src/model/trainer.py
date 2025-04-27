import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config.training_config import training_config
from config.model_config import model_config

class ModelTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config["training_parameters"]["learning_rate"],
            weight_decay=training_config["training_parameters"]["weight_decay"]
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(
                epoch / training_config["training_parameters"]["warmup_steps"], 
                1.0
            )
        )
        
    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            inputs = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                training_config["training_parameters"]["max_grad_norm"]
            )
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
