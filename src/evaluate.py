import torch
from model import TransformerModel
from utils.data_loader import BibleDataset
from utils.logger import setup_logger
from config import model_config

def evaluate_model(model_path, test_texts, test_labels):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("logs/evaluation")
    
    # Load model
    model = TransformerModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Prepare dataset
    dataset = BibleDataset(
        test_texts, 
        test_labels,
        tokenizer_name=model_config["tokenizer_name"],
        max_length=model_config["max_seq_length"]
    )
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Evaluation
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    logger.info(f"Evaluation Results - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    return {
        "loss": avg_loss,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    evaluate_model(
        model_path="path/to/model.pt",
        test_texts=...,
        test_labels=...
    )
