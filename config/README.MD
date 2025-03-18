# Configuration Files

Welcome to the `config/` directory of the **Bible-AI** project! This folder contains all configuration files necessary to customize the behavior of the application, from data processing and model training to frontend rendering and security settings. These files are written in JSON format and are designed to be modular, extensible, and easy to maintain.

## Overview

The Bible-AI project relies on a configuration-driven approach to ensure flexibility and consistency across its components. Each file in this directory serves a specific purpose, interfacing with modules in `src/`, `frontend/`, and `deploy/`. Below is a list of the configuration files, their roles, and how they integrate with the project.

### Files

| File                      | Purpose                                                                         | Used By                                     |
| ------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| `model_config.json`       | Defines the architecture and hyperparameters of the `BiblicalTransformer` model | `src/model/architecture.py`                 |
| `training_config.json`    | Specifies training parameters (e.g., epochs, learning rate)                     | `src/training/trainer.py`                   |
| `data_config.json`        | Configures data processing pipelines (e.g., preprocessing, tokenization)        | `src/data/*.py`                             |
| `data_config_schema.json` | JSON schema for validating `data_config.json`                                   | `src/data/preprocessing.py`                 |
| `theological_rules.json`  | Sets rules for theological validation of model outputs                          | `src/theology/validator.py`                 |
| `frontend_config.json`    | Configures the frontend application (e.g., API endpoints, themes)               | `frontend/src/index.js`                     |
| `bible_sources.json`      | Lists sources for Bible texts and commentaries                                  | `src/bible_manager/downloader.py`           |
| `security_config.json`    | Defines security settings (e.g., authentication, rate limiting)                 | `src/serve/api.py`, `src/utils/security.py` |
| `logging_config.json`     | Configures logging across the application                                       | `src/utils/logger.py`                       |

## Getting Started

### Prerequisites
- Ensure all required dependencies are installed from `requirements.txt`:
  ```bash
  pip install -r requirements.txt