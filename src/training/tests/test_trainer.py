import pytest
import torch
from src.training.trainer import Trainer

@pytest.fixture
def mock_config(tmp_path):
    config_path = tmp_path / "training_config.json"
    config = {
        "model_params": {"hidden_size": 64, "num_hidden_layers": 2},
        "data": {"data_path": "tests/data"},
        "training": {"batch_size": 2, "max_epochs": 1},
        "optimizer": {"learning_rate": 1e-4},
        "logging": {"log_dir": str(tmp_path / "logs")}
    }
    config_path.write_text(json.dumps(config))
    return str(config_path)

def test_trainer_init(mock_config, mocker):
    mocker.patch("src.data.preprocessing.load_processed_data", return_value=([torch.zeros(4, 10)], [torch.zeros(2, 10)]))
    mocker.patch("src.model.architecture.BiblicalTransformer")
    trainer = Trainer(mock_config)
    assert trainer.device.type in ["cuda", "cpu"]