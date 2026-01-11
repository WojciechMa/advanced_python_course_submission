# Data Directory

This directory is for storing medical imaging data files.

## Expected Structure

```
data/
├── dicom/           # DICOM image files
│   ├── patient_001/
│   │   └── *.dcm
│   └── patient_002/
│       └── *.dcm
├── lab_tests/       # Clinical lab test data
│   └── blood_tests.csv
└── processed/       # Preprocessed outputs
    └── (generated)
```

## Getting Sample Data

### Option 1: Generate Synthetic Data
Use the synthetic data generator from the main project:
```bash
python ../scripts/generate_synthetic_data.py
```

### Option 2: Public DICOM Datasets
- [NIH Chest X-rays](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- [Cancer Imaging Archive](https://www.cancerimagingarchive.net/)

### Option 3: DICOM Sample Files
- [pydicom test files](https://github.com/pydicom/pydicom/tree/main/pydicom/data/test_files)

## Data Format

### DICOM Files
- Extension: `.dcm`
- Contains: Image pixels + metadata (patient info, study date, etc.)
- Modalities: CT, MR, CR, DX

### Blood Test CSV
Expected columns:
```csv
patient_id,test_name,value,unit,test_date
P001,WBC,7.5,K/uL,2024-01-15
P001,Hemoglobin,14.2,g/dL,2024-01-15
```

## ⚠️ Important Notes

1. **Never commit real patient data to git!**
2. Add `data/` to `.gitignore`
3. Use anonymized or synthetic data for development
4. Real DICOM files contain PHI (Protected Health Information)
