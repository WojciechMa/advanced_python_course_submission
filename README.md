# Medical Imaging DICOM Processing Pipeline

A Python package for processing, analyzing, and visualizing medical imaging data (DICOM format) with clinical lab test correlation.

## 📋 Overview

This project demonstrates advanced Python programming concepts through a medical imaging analysis pipeline that:

1. **Loads and validates DICOM files** - Medical imaging standard format
2. **Preprocesses images** - Normalization, resizing, windowing
3. **Anonymizes patient data** - HIPAA compliance through PHI removal
4. **Processes clinical lab data** - Blood test loading with pandas
5. **Visualizes results** - Charts, timelines, and reports

## 🎯 Key Features

| Module | Description | Libraries Used |
|--------|-------------|----------------|
| `ingestion/` | DICOM file loading, validation, anonymization | pydicom, numpy |
| `preprocessing/` | Image processing pipeline | numpy, PIL, scikit-learn |
| `visualization/` | Data visualization and reporting | matplotlib, seaborn, pandas |
| `utils/` | Configuration and logging utilities | pyyaml, logging |

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.ingestion import DICOMLoader, Anonymizer
from src.preprocessing import ImagePreprocessor
from src.visualization import ResultsVisualizer

# Load DICOM files
loader = DICOMLoader()
dicom_data = loader.load_directory("data/dicom_samples")

# Anonymize patient data
anonymizer = Anonymizer(anonymization_level="strict")
anon_data = anonymizer.anonymize_dicom_dict(dicom_data[0])

# Preprocess images
preprocessor = ImagePreprocessor(target_size=(224, 224))
processed = preprocessor.preprocess_image("path/to/file.dcm")

# Visualize results
visualizer = ResultsVisualizer()
visualizer.plot_prediction_distribution(predictions)
```

### Run Demo

```bash
python scripts/run_demo.py
```

Or explore the Jupyter notebook:
```bash
jupyter notebook notebooks/dicom_exploration.ipynb
```

## 📁 Project Structure

```
course_submission/
├── README.md                 # This file
├── CONTEXT.md               # Project context and background
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py
│   ├── ingestion/           # Data loading modules
│   │   ├── dicom_loader.py      # DICOM file loading
│   │   ├── dicom_validator.py   # Format validation
│   │   ├── anonymizer.py        # PHI anonymization
│   │   ├── metadata_extractor.py # Metadata extraction
│   │   └── blood_test_loader.py # Lab data (pandas)
│   ├── preprocessing/       # Image processing
│   │   └── image_preprocessor.py
│   ├── visualization/       # Charts and reports
│   │   └── visualizer.py
│   └── utils/              # Utilities
│       ├── logging_config.py
│       └── config_loader.py
├── scripts/
│   └── run_demo.py         # Demo script
├── notebooks/
│   └── dicom_exploration.ipynb  # Interactive demo
├── tests/                  # Unit tests
│   └── test_ingestion.py
├── config/
│   └── pipeline_config.yaml
└── data/
    └── README.md           # Data instructions
```

## 🔬 Technical Highlights

### Object-Oriented Programming
- Classes with proper encapsulation
- Type hints throughout
- Comprehensive docstrings

### NumPy Operations
- Image array manipulation
- Windowing and normalization
- Batch processing

### Pandas DataFrames
- Blood test data loading
- Unit conversions
- Reference range validation

### File I/O
- DICOM file parsing with pydicom
- JSON/YAML configuration
- CSV data loading

### Visualization
- Matplotlib/Seaborn charts
- Timeline visualizations
- Correlation heatmaps
- HTML report generation

## 📊 Example Outputs

### DICOM Loading Statistics
```
Loaded 50 DICOM files successfully
- Successful: 48
- Failed: 2
- Skipped (unsupported modality): 0
```

### Preprocessing Pipeline
```
Input: chest_xray_001.dcm
  Original size: (2048, 2048)
  Output size: (224, 224)
  Normalization: [0, 1]
  Processing time: 0.15s
```

### Blood Test Analysis
```
Loaded 150 lab test records
- Normal values: 120 (80%)
- Abnormal HIGH: 18 (12%)
- Abnormal LOW: 12 (8%)
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 Dependencies

- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **pydicom** - DICOM file handling
- **matplotlib** - Plotting
- **seaborn** - Statistical visualization
- **scikit-learn** - Dataset splitting
- **pillow** - Image processing
- **pyyaml** - Configuration files

## 📖 DICOM Background

DICOM (Digital Imaging and Communications in Medicine) is the standard for medical imaging data. Key concepts:

- **Modalities**: CT, MR, CR (X-ray), DX (Digital X-ray)
- **Windowing**: Adjusting contrast for visualization
- **PHI**: Protected Health Information (must be anonymized)

## 🔗 Related Resources

- [DICOM Standard](https://www.dicomstandard.org/)
- [pydicom Documentation](https://pydicom.github.io/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 📝 License

MIT License - See LICENSE file for details.

## 👤 Author

Wojciech Majek - Advanced Python Course (Cognitive Science course. Faculty of Psychology, University of Warsaw)

---

*This is a course submission extract from a larger Medical Imaging Analysis project.*
