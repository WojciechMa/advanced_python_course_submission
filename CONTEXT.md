# Course Submission Context

## What Is This?

This is a **standalone extract** from a larger Medical Imaging Analysis project. It contains the DICOM processing, preprocessing, and visualization components - designed for an Advanced Python course submission.

## Module Overview

```
src/
├── ingestion/          # DICOM file handling
│   ├── dicom_loader.py       - Load DICOM files
│   ├── dicom_validator.py    - Validate DICOM compliance
│   ├── anonymizer.py         - Remove patient identifiers
│   ├── metadata_extractor.py - Extract DICOM metadata
│   └── blood_test_loader.py  - Load clinical lab data (pandas)
│
├── preprocessing/      # Image processing
│   └── image_preprocessor.py - Resize, normalize, augment images
│
├── visualization/      # Data visualization
│   └── visualizer.py         - Charts, timelines, heatmaps
│
└── utils/              # Utilities
    ├── logging_config.py     - Structured logging
    └── config_loader.py      - YAML configuration
```

## Key Skills Demonstrated

| Skill | Where to Find |
|-------|---------------|
| **NumPy arrays** | `preprocessing/image_preprocessor.py` |
| **Pandas DataFrames** | `ingestion/blood_test_loader.py` |
| **OOP (Classes)** | All modules |
| **File I/O** | `ingestion/dicom_loader.py` |
| **Matplotlib/Seaborn** | `visualization/visualizer.py` |
| **Configuration** | `utils/config_loader.py` |
| **Logging** | `utils/logging_config.py` |
| **Type hints** | All modules |
| **Docstrings** | All modules |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo script
python scripts/run_demo.py

# Or explore the notebook
jupyter notebook notebooks/dicom_exploration.ipynb
```

## Relationship to Full Project

This is extracted from a larger project that includes:
- Deep learning models (PyTorch CNNs)
- REST API (FastAPI)
- Database integration (PostgreSQL)
- ML experiment tracking (MLflow)
- Distributed processing (PySpark)

Those components will be developed further in ML courses next semester.

## Data

The `data/` folder should contain DICOM files for processing. Sample synthetic data can be generated - see main project's `scripts/generate_synthetic_data.py`.
