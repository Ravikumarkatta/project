
### Checklist to Start Training the Bible-AI Model

#### 1. Environment Setup
- [ ] **Python Version**: Ensure Python 3.8+ is installed.
  - Verify: `python --version`.
- [ ] **Dependencies**: Install all required packages.
  - Run: `pip install -r requirements.txt`.
  - Key packages: `torch`, `transformers`, `fastapi`, `redis`, `pydantic`, etc.
- [ ] **GPU Support** (Optional but recommended):
  - Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`.
  - Install CUDA-enabled PyTorch if needed: Adjust `requirements.txt` or use `pip install torch -f https://download.pytorch.org/whl/cu{version}/torch_stable.html`.
- [ ] **Redis**: For caching in `src/serve/cache.py`.
  - Install: `sudo apt install redis-server` (Ubuntu) or equivalent.
  - Start: `redis-server &`.
  - Verify: `redis-cli ping` (returns "PONG").
- [ ] **Node.js** (For frontend, optional for training):
  - Verify: `node --version` (16+).
  - Install frontend deps if testing UI: `cd frontend && npm install`.
- [ ] **Working Directory**: Ensure you’re in `bible-ai/`.
  - Verify: `pwd` ends with `/bible-ai`.

#### 2. Data Preparation
- [ ] **Raw Data Sources**:
  - **Bible Translations**: Populate `data/raw/bibles/`.
    - Run: `python src/bible_manager/downloader.py --versions NIV,ESV,KJV` (or use `cli.py download-bible` if implemented).
    - Verify: Check `data/raw/bibles/` for files (e.g., `NIV.txt`, `ESV.json`).
  - **Commentaries**: Add commentary texts to `data/raw/commentaries/`.
    - Source manually or script via `downloader.py`.
    - Verify: `ls data/raw/commentaries/`.
  - **QA Pairs**: Add question-answer pairs to `data/raw/qa_pairs/`.
    - Example: `qa_pairs.json` with `{"question": "What is love?", "answer": "John 3:16"}`.
    - Verify: Check directory.
  - **Theological Data**: Ensure `data/raw/theological/technical_terms.json` exists.
    - Verify: `cat data/raw/theological/technical_terms.json`.
  - **Lexicons**: Populate `data/raw/lexicons/hebrew/` and `data/raw/lexicons/greek/`.
    - Source from external lexicon datasets (e.g., Strong’s).
    - Verify: Check subdirectories.
- [ ] **Processed Data**:
  - Generate datasets: `python scripts/generate_dataset.py`.
    - Ensure `src/data/preprocessing.py`, `tokenization.py`, and `dataset.py` use `data_config.json`.
  - Verify: Check `data/processed/` for files (e.g., `train.pt`, `valid.pt`).
- [ ] **Embeddings** (Optional):
  - Pre-compute embeddings if needed: Update `src/data/dataset.py`.
  - Verify: Check `data/embeddings/` for `.pt` or `.npy` files.
- [ ] **Uploads**: If using custom texts, place them in `data/uploads/`.
  - Process with `src/bible_manager/uploader.py`.
  - Verify: `ls data/uploads/`.

#### 3. Configuration Files
- [ ] **Model Configuration (`config/model_config.json`)**:
  - Verify or create:
    ```json
    {
      "vocab_size": 30522,  // Adjust based on tokenizer
      "hidden_size": 768,
      "num_layers": 12,
      "num_heads": 12,
      "max_position_embeddings": 512,
      "dropout": 0.1,
      "model_type": "transformer"
    }
    ```
  - Ensure `src/model/architecture.py` reads this.
- [ ] **Training Configuration (`config/training_config.json`)**:
  - Verify or create:
    ```json
    {
      "batch_size": 16,
      "num_epochs": 10,
      "learning_rate": 5e-5,
      "warmup_steps": 1000,
      "max_grad_norm": 1.0,
      "save_steps": 1000,
      "output_dir": "data/snapshots",
      "device": "cuda",  // "cpu" if no GPU
      "validation_interval": 500
    }
    ```
  - Create `data/snapshots/`: `mkdir -p data/snapshots`.
- [ ] **Data Configuration (`config/data_config.json`)**:
  - Verify paths:
    ```json
    {
      "raw_dir": "data/raw",
      "processed_dir": "data/processed",
      "uploads_dir": "data/uploads",
      "embeddings_dir": "data/embeddings",
      "snapshots_dir": "data/snapshots",
      "preprocessing": {
        "remove_punctuation": false,
        "normalize_case": true,
        "strip_html": true
      },
      "tokenization": {
        "max_length": 512
      }
    }
    ```
  - Validate: `python -c "from jsonschema import validate; import json; with open('config/data_config_schema.json') as s, open('config/data_config.json') as c: validate(json.load(c), json.load(s))"`.
- [ ] **Theological Rules (`config/theological_rules.json`)**:
  - Verify existence (crafted earlier).
- [ ] **Logging (`config/logging_config.json`)**:
  - Verify existence and `logs/` directory: `mkdir -p logs`.

#### 4. Code Verification
- [ ] **Model (`src/model/architecture.py`)**:
  - Ensure `BiblicalTransformer` initializes with `model_config.json`.
  - Verify tokenizer integration with `src/data/tokenization.py`.
- [ ] **Training (`src/training/trainer.py`)**:
  - Ensure it loads `training_config.json` and integrates `TheologicalValidator`:
    ```python
    from src.theology import TheologicalValidator
    validator = TheologicalValidator()
    scores = validator.validate({"text": predicted_text})
    ```
  - Verify `loss.py`, `optimization.py`, and `checkpointing.py` are implemented.
- [ ] **Data Pipeline**:
  - Check `src/data/preprocessing.py`, `tokenization.py`, `augmentation.py`, and `dataset.py`.
  - Test: `python -c "from src.data.dataset import BibleDataset; print(BibleDataset('data/processed/train.pt'))"`.
- [ ] **Theology (`src/theology/`)**:
  - Verify all files (`validator.py`, `doctrines.py`, etc.) load `theological_rules.json`.
- [ ] **Monitoring (`src/monitoring/metrics.py`)**:
  - Ensure it tracks training metrics.

#### 5. Hardware Check
- [ ] **RAM**: 8GB+ recommended.
  - Verify: `free -h` (Linux).
- [ ] **Disk Space**: 10GB+ free for snapshots.
  - Verify: `df -h`.
- [ ] **GPU** (If available):
  - Check: `nvidia-smi` (ensure 4GB+ VRAM).

#### 6. Pre-Training Steps
- [ ] **Data Integrity**:
  - Run: `python scripts/verify_biblical_data.py`.
  - Check logs: `cat logs/preprocessing.log`.
- [ ] **Tokenizer Test**:
  - Run: `python -c "from src.data.tokenization import BiblicalTokenizer; t = BiblicalTokenizer(); print(t.tokenize('In the beginning God created'))"`.
- [ ] **Dry Run**:
  - Test: `python cli.py train-model --config config/training_config.json --dry-run`.
  - Verify logs: `cat logs/training.log`.

#### 7. Initiate Training
- [ ] **Command**:
  - Run: `python cli.py train-model --config config/training_config.json`.
  - Optional: `--resume-from data/snapshots/checkpoint.pt`.
- [ ] **Monitor**:
  - Tail logs: `tail -f logs/training.log`.
  - Check `data/snapshots/` for checkpoints.
- [ ] **Adjustments**:
  - If OOM error, reduce `batch_size` or use gradient accumulation in `trainer.py`.

#### 8. Post-Training Validation
- [ ] **Evaluate**:
  - Run: `python scripts/evaluate_model.py --model data/snapshots/best_model.pt`.
- [ ] **Theological Check**:
  - Run: `python scripts/theology_validator.py --model data/snapshots/best_model.pt`.
- [ ] **API Test**:
  - Start: `python src/serve/api.py`.
  - Test: `curl -X POST "http://localhost:8000/generate" -H "Authorization: Bearer <token>" -d '{"text": "What is faith?"}'`.

---

### Adjustments for Project Structure

Since `bible.txt` isn’t a specific file but part of the ecosystem:
- **Data Sources**: Training will use `data/raw/bibles/`, `data/raw/commentaries/`, `data/raw/qa_pairs/`, and `data/raw/theological/`.
- **Processing**: `scripts/generate_dataset.py` should aggregate these into `data/processed/` datasets.
- **Tokenizer**: `src/data/tokenization.py` must handle biblical text, commentary, and QA formats.

---

### Sample `cli.py` for Training

If not fully implemented:
```python
# cli.py (snippet)
import argparse
from src.training.trainer import Trainer

def train_model(args):
    trainer = Trainer(config_path=args.config)
    trainer.train()

parser = argparse.ArgumentParser(description="Bible-AI CLI")
subparsers = parser.add_subparsers(dest="command")
train_parser = subparsers.add_parser("train-model", help="Train the model")
train_parser.add_argument("--config", required=True, help="Training config path")
train_parser.add_argument("--dry-run", action="store_true", help="Dry run without training")
train_parser.add_argument("--resume-from", help="Resume from checkpoint")
args = parser.parse_args()

if args.command == "train-model":
    train_model(args)
```

---

### Troubleshooting

- **Missing Data**: Populate `data/raw/` subdirectories.
- **Config Errors**: Validate JSON with schemas (e.g., `data_config_schema.json`).
- **Slow Training**: Switch to CPU or optimize `batch_size`.

---

This checklist leverages the full `bible-ai/` structure, ensuring a robust training process. Let me know if you need help implementing any missing files (e.g., `trainer.py`) or running the training!