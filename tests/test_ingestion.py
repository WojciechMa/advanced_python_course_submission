"""Basic tests for the ingestion module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import DICOMLoader, DICOMValidator, Anonymizer, MetadataExtractor, BloodTestLoader


class TestDICOMLoader:
    """Tests for DICOMLoader class."""
    
    def test_initialization(self):
        """Test DICOMLoader initializes with correct defaults."""
        loader = DICOMLoader()
        
        assert loader.source_type == "local"
        assert loader.batch_size == 100
        assert "CT" in loader.supported_modalities
        assert "DX" in loader.supported_modalities
    
    def test_custom_initialization(self):
        """Test DICOMLoader with custom parameters."""
        loader = DICOMLoader(
            source_type="local",
            batch_size=50,
            supported_modalities=["CT", "MR"]
        )
        
        assert loader.batch_size == 50
        assert loader.supported_modalities == ["CT", "MR"]
    
    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        loader = DICOMLoader()
        loader.stats["successful_loads"] = 10
        
        loader.reset_statistics()
        
        assert loader.stats["successful_loads"] == 0
        assert loader.stats["failed_loads"] == 0


class TestDICOMValidator:
    """Tests for DICOMValidator class."""
    
    def test_initialization(self):
        """Test DICOMValidator initializes correctly."""
        validator = DICOMValidator()
        
        assert "PatientID" in validator.required_tags
        assert "Modality" in validator.required_tags
    
    def test_modality_validation(self):
        """Test modality validation logic."""
        validator = DICOMValidator()
        
        # Valid modalities
        assert validator._is_valid_modality("CT") == True
        assert validator._is_valid_modality("MR") == True
        assert validator._is_valid_modality("DX") == True
        
        # Invalid modalities
        assert validator._is_valid_modality("INVALID") == False
        assert validator._is_valid_modality("123") == False
    
    def test_date_validation(self):
        """Test DICOM date format validation."""
        validator = DICOMValidator()
        
        # Valid dates (YYYYMMDD format)
        assert validator._is_valid_date("20240115") == True
        assert validator._is_valid_date("19990101") == True
        
        # Invalid dates
        assert validator._is_valid_date("2024-01-15") == False  # Wrong format
        assert validator._is_valid_date("invalid") == False
        assert validator._is_valid_date("") == False


class TestAnonymizer:
    """Tests for Anonymizer class."""
    
    def test_initialization(self):
        """Test Anonymizer initializes correctly."""
        anonymizer = Anonymizer()
        
        assert anonymizer.anonymization_level == "strict"
        assert anonymizer.date_shift_days == 0
        assert len(anonymizer.PHI_TAGS) > 0
    
    def test_patient_id_hashing(self):
        """Test patient ID hashing produces consistent results."""
        anonymizer = Anonymizer()
        
        # Same input should produce same output
        hash1 = anonymizer.hash_patient_id("PATIENT_001")
        hash2 = anonymizer.hash_patient_id("PATIENT_001")
        
        assert hash1 == hash2
        assert hash1.startswith("ANON_")
    
    def test_different_ids_produce_different_hashes(self):
        """Test different patient IDs produce different hashes."""
        anonymizer = Anonymizer()
        
        hash1 = anonymizer.hash_patient_id("PATIENT_001")
        hash2 = anonymizer.hash_patient_id("PATIENT_002")
        
        assert hash1 != hash2
    
    def test_date_shifting(self):
        """Test date shifting functionality."""
        anonymizer = Anonymizer(date_shift_days=30)
        
        original = "2024-01-15"
        shifted = anonymizer._shift_date_string(original)
        
        assert shifted == "2024-02-14"  # 30 days later
    
    def test_id_mapping_tracking(self):
        """Test that ID mappings are tracked."""
        anonymizer = Anonymizer()
        
        anonymizer.hash_patient_id("PATIENT_001")
        anonymizer.hash_patient_id("PATIENT_002")
        
        mapping = anonymizer.get_id_mapping()
        
        assert len(mapping) == 2
        assert "PATIENT_001" in mapping
        assert "PATIENT_002" in mapping


class TestBloodTestLoader:
    """Tests for BloodTestLoader class."""
    
    def test_initialization(self):
        """Test BloodTestLoader initializes correctly."""
        loader = BloodTestLoader()
        
        assert loader.normalize_units == True
        assert loader.add_reference_ranges == True
        assert loader.validate_values == True
    
    def test_dataframe_processing(self):
        """Test DataFrame processing adds expected columns."""
        loader = BloodTestLoader()
        
        df = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'lab_name': ['WBC', 'Hemoglobin'],
            'value': [7.5, 14.0],
            'unit': ['K/uL', 'g/dL']
        })
        
        processed = loader.load_dataframe(df)
        
        # Check reference ranges were added
        assert 'ref_min' in processed.columns
        assert 'ref_max' in processed.columns
        
        # Check validation was performed
        assert 'is_abnormal' in processed.columns
        assert 'abnormal_flag' in processed.columns
    
    def test_abnormal_detection(self):
        """Test abnormal value detection."""
        loader = BloodTestLoader()
        
        # WBC reference range is 4.5-11.0 K/uL
        df = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'lab_name': ['WBC', 'WBC', 'WBC'],
            'value': [7.5, 2.0, 15.0],  # Normal, Low, High
            'unit': ['K/uL', 'K/uL', 'K/uL']
        })
        
        processed = loader.load_dataframe(df)
        
        flags = processed['abnormal_flag'].tolist()
        
        assert flags[0] == 'NORMAL'
        assert flags[1] == 'LOW'
        assert flags[2] == 'HIGH'


class TestMetadataExtractor:
    """Tests for MetadataExtractor class."""
    
    def test_initialization(self):
        """Test MetadataExtractor initializes correctly."""
        extractor = MetadataExtractor()
        
        assert "PatientID" in extractor.required_tags
        assert "Modality" in extractor.required_tags
    
    def test_custom_tags(self):
        """Test adding custom tags."""
        extractor = MetadataExtractor(custom_tags=["CustomTag1", "CustomTag2"])
        
        assert "CustomTag1" in extractor.optional_tags
        assert "CustomTag2" in extractor.optional_tags


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
