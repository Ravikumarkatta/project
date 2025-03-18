# Bible-AI Architecture

This document provides an overview of the Bible-AI system architecture for developers interested in understanding or contributing to the project.

## System Overview

Bible-AI is a full-stack application with the following major components:

1. **AI Model Core**: Custom transformer-based models for biblical understanding
2. **Data Processing Pipeline**: Bible text processing and knowledge base creation
3. **Theological Validation System**: Ensures responses align with sound interpretation
4. **Backend API**: Serves model outputs and manages resources
5. **Frontend Application**: React-based user interface

## Architecture Diagram

```
┌────────────────┐     ┌─────────────────┐     ┌────────────────┐
│                │     │                 │     │                │
│  Bible Sources ├────►│  Data Pipeline  ├────►│  Training Data │
│                │     │                 │     │                │
└────────────────┘     └─────────────────┘     └───────┬────────┘
                                                       │
                                                       ▼
┌────────────────┐     ┌─────────────────┐     ┌───────────────┐
│                │     │                 │     │               │
│   Frontend     │◄───►│   Backend API   │◄───►│   AI Models   │
│                │     │                 │     │               │
└────────────────┘     └─────────────────┘     └───────┬───────┘
                                                       │
                                                       ▼
                                               ┌───────────────┐
                                               │  Theological  │
                                               │  Validation   │
                                               └───────────────┘
```

## Core Components

### Data Processing (src/data/)

The data processing pipeline handles:
- Bible text acquisition and normalization
- Commentary processing and indexing
- Lexical data (Hebrew/Greek) preparation
- Training dataset creation
- Data augmentation for theological concepts

Key files:
- `preprocessing.py`: Text cleaning and normalization
- `tokenization.py`: Custom tokenization for biblical content
- `dataset.py`: Dataset creation and management

### Bible Management (src/bible_manager/)

Manages various Bible resources:
- Downloading and parsing Bible translations
- Handling file formats (XML, USFM, JSON)
- Verse reference resolution and validation
- Text indexing for search

### Model Architecture (src/model/)

The AI model architecture includes:
- Custom transformer-based architecture
- Specialized attention mechanisms for biblical cross-references
- Multi-task learning capabilities
- Biblical reference detection and resolution

### Theological Validation (src/theology/)

The theological validation system:
- Ensures responses align with Scripture
- Manages denominational perspectives
- Handles controversial topics appropriately
- Provides pastoral sensitivity

Implementation details:
- Rule-based validation system
- Denominational configuration options
- Biblical reference verification
- Content sensitivity checks

### Backend API (src/serve/)

The API layer provides:
- Model inference endpoints
- Resource management
- Authentication and authorization
- Request validation
- Caching for performance

### Frontend Application (frontend/)

The React-based frontend offers:
- User interface for Bible study
- Chat interface for theological questions
- Bible text display and navigation
- Study tools (notes, highlights, word study)
- User preference management

## Data Flow

1. **Input Processing**:
   - User queries are processed and tokenized
   - Bible references are identified and resolved
   - Query intent is classified (study, question, exploration)

2. **Model Inference**:
   - Appropriate model is selected based on query type
   - Contextual information is retrieved
   - Response is generated

3. **Theological Validation**:
   - Response is checked against theological validation rules
   - Denominational context is applied if specified
   - Biblical references are verified

4. **Response Delivery**:
   - Formatted response is sent to frontend
   - References are linked to Bible text
   - Additional resources are included when relevant

## Configuration System

Bible-AI uses a comprehensive configuration system:
- `model_config.json`: Model architecture parameters
- `training_config.json`: Training hyperparameters
- `data_config.json`: Data processing settings
- `theological_rules.json`: Theological validation rules
- `frontend_config.json`: UI configuration

## Extending the Architecture

Developers can extend Bible-AI by:
1. Adding new model architectures in `/src/model/`
2. Implementing additional theological validators in `/src/theology/`
3. Creating new data processors in `/src/data/`
4. Expanding frontend components in `/frontend/src/components/`

See `extending_model.md` for detailed implementation guides.
### Tokenization
The `tokenization.py` module handles custom tokenization for biblical texts:
```python
from src.data.tokenization import BiblicalTokenizer
tokenizer = BiblicalTokenizer()
encoding = tokenizer.tokenize("John 3:16 For God so loved the world.")
