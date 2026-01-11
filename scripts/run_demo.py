#!/usr/bin/env python
"""
Demo script for Medical Imaging DICOM Processing Pipeline.

This script demonstrates the main functionality of the package:
1. DICOM file loading and validation
2. Patient data anonymization
3. Image preprocessing
4. Blood test data loading with pandas
5. Visualization

Run with: python scripts/run_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def demo_dicom_loader():
    """Demonstrate DICOM loading functionality."""
    print_header("1. DICOM LOADER DEMO")
    
    from ingestion import DICOMLoader
    
    # Initialize loader
    loader = DICOMLoader(
        source_type="local",
        batch_size=100,
        anonymization_level="strict",
        supported_modalities=["CT", "MR", "CR", "DX"]
    )
    
    print_subheader("Loader Configuration")
    print(f"  Source type: {loader.source_type}")
    print(f"  Batch size: {loader.batch_size}")
    print(f"  Supported modalities: {loader.supported_modalities}")
    
    # Check for sample data
    data_dir = Path(__file__).parent.parent / "data"
    dicom_dir = data_dir / "dicom"
    
    if dicom_dir.exists():
        dcm_files = list(dicom_dir.rglob("*.dcm"))
        if dcm_files:
            print_subheader(f"Loading {len(dcm_files)} DICOM files")
            results = loader.load_directory(dicom_dir)
            
            stats = loader.get_statistics()
            print(f"\n  Loading Statistics:")
            print(f"    Total files: {stats['total_files']}")
            print(f"    Successful: {stats['successful_loads']}")
            print(f"    Failed: {stats['failed_loads']}")
            print(f"    Skipped: {stats['skipped_files']}")
            
            if results:
                print(f"\n  Sample metadata from first file:")
                sample = results[0]['metadata']
                for key in ['Modality', 'StudyDate', 'Rows', 'Columns']:
                    if key in sample:
                        print(f"    {key}: {sample[key]}")
        else:
            print(f"\n  No DICOM files found in {dicom_dir}")
            print("  Place .dcm files in data/dicom/ to test loading")
    else:
        print(f"\n  Data directory not found: {dicom_dir}")
        print("  See data/README.md for instructions")
    
    return loader


def demo_validator():
    """Demonstrate DICOM validation functionality."""
    print_header("2. DICOM VALIDATOR DEMO")
    
    from ingestion import DICOMValidator
    
    # Initialize validator
    validator = DICOMValidator(
        required_tags=["PatientID", "StudyDate", "Modality", "SOPInstanceUID"]
    )
    
    print_subheader("Validator Configuration")
    print(f"  Required tags: {validator.required_tags}")
    
    # Test validation methods
    print_subheader("Testing Validation Logic")
    
    # Test modality validation
    valid_modalities = ["CT", "MR", "CR", "DX"]
    invalid_modalities = ["INVALID", "123"]
    
    print("\n  Modality validation:")
    for mod in valid_modalities[:2] + invalid_modalities[:1]:
        result = validator._is_valid_modality(mod)
        status = "✓ Valid" if result else "✗ Invalid"
        print(f"    {mod}: {status}")
    
    # Test date validation
    print("\n  Date validation:")
    test_dates = ["20240115", "2024-01-15", "invalid"]
    for date in test_dates:
        result = validator._is_valid_date(date)
        status = "✓ Valid" if result else "✗ Invalid"
        print(f"    '{date}': {status}")
    
    return validator


def demo_anonymizer():
    """Demonstrate anonymization functionality."""
    print_header("3. ANONYMIZER DEMO")
    
    from ingestion import Anonymizer
    
    # Initialize anonymizer
    anonymizer = Anonymizer(
        anonymization_level="strict",
        date_shift_days=30
    )
    
    print_subheader("Anonymizer Configuration")
    print(f"  Anonymization level: {anonymizer.anonymization_level}")
    print(f"  Date shift: {anonymizer.date_shift_days} days")
    print(f"  PHI tags to remove: {len(anonymizer.PHI_TAGS)}")
    
    # Demonstrate patient ID hashing
    print_subheader("Patient ID Hashing")
    test_ids = ["PATIENT_001", "PATIENT_002", "PATIENT_001"]  # Note: duplicate
    
    print("\n  Original ID -> Anonymized ID:")
    for pid in test_ids:
        anon_id = anonymizer.hash_patient_id(pid)
        print(f"    {pid} -> {anon_id}")
    
    print("\n  Note: Same patient ID always produces same hash (for data linkage)")
    
    # Demonstrate date shifting
    print_subheader("Date Shifting")
    test_date = "2024-01-15"
    shifted = anonymizer._shift_date_string(test_date)
    print(f"\n  Original: {test_date}")
    print(f"  Shifted (+{anonymizer.date_shift_days} days): {shifted}")
    
    return anonymizer


def demo_metadata_extractor():
    """Demonstrate metadata extraction functionality."""
    print_header("4. METADATA EXTRACTOR DEMO")
    
    from ingestion import MetadataExtractor
    
    # Initialize extractor
    extractor = MetadataExtractor(custom_tags=["ImageComments"])
    
    print_subheader("Extractor Configuration")
    print(f"  Required tags: {extractor.required_tags}")
    print(f"  Optional tags: {len(extractor.optional_tags)} tags configured")
    
    print_subheader("Key Metadata Categories")
    categories = {
        "Patient Info": ["PatientID", "PatientName", "PatientBirthDate", "PatientSex"],
        "Study Info": ["StudyDate", "StudyDescription", "Modality"],
        "Image Info": ["Rows", "Columns", "PixelSpacing", "WindowCenter"]
    }
    
    for cat, tags in categories.items():
        print(f"\n  {cat}:")
        for tag in tags:
            print(f"    - {tag}")
    
    return extractor


def demo_blood_test_loader():
    """Demonstrate blood test loading functionality."""
    print_header("5. BLOOD TEST LOADER DEMO")
    
    from ingestion import BloodTestLoader
    from ingestion.blood_test_loader import REFERENCE_RANGES, UNIT_CONVERSIONS
    
    # Initialize loader
    loader = BloodTestLoader(
        normalize_units=True,
        add_reference_ranges=True,
        validate_values=True
    )
    
    print_subheader("Loader Configuration")
    print(f"  Normalize units: {loader.normalize_units}")
    print(f"  Add reference ranges: {loader.add_reference_ranges}")
    print(f"  Validate values: {loader.validate_values}")
    
    # Show reference ranges
    print_subheader("Reference Ranges (Sample)")
    for test_name in list(REFERENCE_RANGES.keys())[:5]:
        ref = REFERENCE_RANGES[test_name]
        print(f"  {test_name}: {ref['min']}-{ref['max']} {ref['unit']}")
    
    # Create sample data with pandas
    print_subheader("Processing Sample Data")
    
    sample_data = pd.DataFrame({
        'patient_id': ['P001', 'P001', 'P001', 'P002', 'P002'],
        'lab_name': ['WBC', 'Hemoglobin', 'Glucose', 'WBC', 'CRP'],
        'value': [7.5, 14.2, 95, 12.5, 45],
        'unit': ['K/uL', 'g/dL', 'mg/dL', 'K/uL', 'mg/L'],
        'test_datetime': pd.date_range('2024-01-15', periods=5, freq='D')
    })
    
    print("\n  Input DataFrame:")
    print(sample_data.to_string(index=False))
    
    # Process the data
    processed = loader.load_dataframe(sample_data)
    
    print("\n  Processed DataFrame (with validation):")
    cols = ['subject_id', 'lab_name', 'value', 'ref_min', 'ref_max', 'abnormal_flag']
    print(processed[cols].to_string(index=False))
    
    # Show statistics
    print_subheader("Value Distribution")
    abnormal_counts = processed['abnormal_flag'].value_counts()
    for flag, count in abnormal_counts.items():
        pct = count / len(processed) * 100
        print(f"  {flag}: {count} ({pct:.0f}%)")
    
    return loader


def demo_preprocessor():
    """Demonstrate image preprocessing functionality."""
    print_header("6. IMAGE PREPROCESSOR DEMO")
    
    from preprocessing import ImagePreprocessor
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        window_center=None,
        window_width=None,
        normalize_method='zero_one',
        augmentation=False,
        random_seed=42
    )
    
    print_subheader("Preprocessor Configuration")
    print(f"  Target size: {preprocessor.target_size}")
    print(f"  Normalize method: {preprocessor.normalize_method}")
    print(f"  Augmentation: {preprocessor.augmentation}")
    
    # Create synthetic image for demo
    print_subheader("Processing Synthetic Image")
    
    # Create a simple test image (gradient)
    test_image = np.linspace(0, 4095, 512*512).reshape(512, 512).astype(np.float32)
    
    print(f"\n  Original image:")
    print(f"    Shape: {test_image.shape}")
    print(f"    Dtype: {test_image.dtype}")
    print(f"    Range: [{test_image.min():.0f}, {test_image.max():.0f}]")
    
    # Apply windowing
    windowed = preprocessor.apply_windowing(test_image)
    print(f"\n  After windowing:")
    print(f"    Shape: {windowed.shape}")
    print(f"    Range: [{windowed.min()}, {windowed.max()}]")
    
    # Resize
    resized = preprocessor.resize_image(windowed)
    print(f"\n  After resizing:")
    print(f"    Shape: {resized.shape}")
    
    # Normalize
    normalized = preprocessor.normalize_image(resized)
    print(f"\n  After normalization ({preprocessor.normalize_method}):")
    print(f"    Shape: {normalized.shape}")
    print(f"    Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Dataset splitting demo
    print_subheader("Dataset Splitting Demo")
    
    fake_paths = [f"image_{i:03d}.dcm" for i in range(100)]
    fake_labels = [0] * 60 + [1] * 40  # 60% class 0, 40% class 1
    
    splits = preprocessor.create_dataset_split(
        fake_paths, fake_labels,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        stratify=True
    )
    
    print(f"\n  Split results (100 samples, stratified):")
    for split_name, data in splits.items():
        labels = data['labels']
        class_0 = labels.count(0)
        class_1 = labels.count(1)
        print(f"    {split_name}: {len(data['paths'])} samples "
              f"(class 0: {class_0}, class 1: {class_1})")
    
    return preprocessor


def demo_visualizer():
    """Demonstrate visualization functionality."""
    print_header("7. VISUALIZER DEMO")
    
    from visualization import ResultsVisualizer
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(
        figure_size=(12, 8),
        style='seaborn-v0_8-darkgrid',
        color_palette='Set2'
    )
    
    print_subheader("Visualizer Configuration")
    print(f"  Figure size: {visualizer.figure_size}")
    print(f"  Style: {visualizer.style}")
    print(f"  Color palette: {visualizer.color_palette}")
    
    print_subheader("Available Visualization Methods")
    methods = [
        ("plot_patient_timeline", "Timeline of imaging + lab tests"),
        ("plot_correlation_heatmap", "Prediction vs biomarker correlations"),
        ("plot_biomarker_trends", "Biomarker values over time"),
        ("plot_prediction_distribution", "Class distribution bar/pie charts"),
        ("plot_confidence_distribution", "Confidence score histograms"),
        ("generate_summary_report", "All visualizations for a patient"),
        ("create_html_report", "Interactive HTML report"),
    ]
    
    for method, desc in methods:
        print(f"  • {method}()")
        print(f"      {desc}")
    
    # Create sample data for visualization
    print_subheader("Sample Visualization Data")
    
    sample_predictions = [
        {"prediction": "Normal", "confidence": 0.92, "study_date": "2024-01-15"},
        {"prediction": "Pneumonia", "confidence": 0.78, "study_date": "2024-02-20"},
        {"prediction": "Normal", "confidence": 0.85, "study_date": "2024-03-10"},
        {"prediction": "Normal", "confidence": 0.95, "study_date": "2024-04-05"},
        {"prediction": "CHF", "confidence": 0.67, "study_date": "2024-05-01"},
    ]
    
    print("\n  Sample predictions:")
    for p in sample_predictions[:3]:
        print(f"    {p['study_date']}: {p['prediction']} ({p['confidence']:.0%})")
    print(f"    ... and {len(sample_predictions)-3} more")
    
    # Save sample visualization if output dir exists
    output_dir = Path(__file__).parent.parent / "output"
    if output_dir.exists() or True:  # Always try
        output_dir.mkdir(exist_ok=True)
        try:
            fig = visualizer.plot_prediction_distribution(
                sample_predictions, 
                save_path=output_dir / "sample_distribution.png"
            )
            print(f"\n  Saved: {output_dir / 'sample_distribution.png'}")
        except Exception as e:
            print(f"\n  Could not save plot: {e}")
    
    return visualizer


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  MEDICAL IMAGING DICOM PROCESSING PIPELINE - DEMO")
    print("  Course Submission: Advanced Python")
    print("=" * 70)
    
    try:
        # Run demos
        demo_dicom_loader()
        demo_validator()
        demo_anonymizer()
        demo_metadata_extractor()
        demo_blood_test_loader()
        demo_preprocessor()
        demo_visualizer()
        
        # Summary
        print_header("DEMO COMPLETE")
        print("\n  All modules demonstrated successfully!")
        print("\n  Key Python concepts used:")
        print("    ✓ Object-Oriented Programming (classes, methods)")
        print("    ✓ NumPy array operations")
        print("    ✓ Pandas DataFrames")
        print("    ✓ Type hints and docstrings")
        print("    ✓ File I/O and path handling")
        print("    ✓ Matplotlib/Seaborn visualization")
        print("    ✓ Configuration with YAML")
        print("    ✓ Logging")
        
        print("\n  Next steps:")
        print("    1. Add DICOM files to data/dicom/")
        print("    2. Explore notebooks/dicom_exploration.ipynb")
        print("    3. Run tests: pytest tests/ -v")
        
    except Exception as e:
        print(f"\n  Error: {e}")
        raise


if __name__ == "__main__":
    main()
