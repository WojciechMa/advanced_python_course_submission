"""DICOM file validation."""

import logging
from typing import Dict, Any, List, Tuple, Optional
import pydicom

logger = logging.getLogger(__name__)


class DICOMValidator:
    """Validate DICOM file format and required fields."""
    
    def __init__(self, required_tags: Optional[List[str]] = None):
        """
        Initialize DICOM validator.
        
        Args:
            required_tags: List of required DICOM tag names
        """
        self.required_tags = required_tags or [
            "PatientID",
            "StudyDate",
            "Modality",
            "SOPInstanceUID",
        ]
        logger.info(f"Initialized DICOMValidator with {len(self.required_tags)} required tags")
    
    def validate(self, dicom_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate DICOM data structure and required fields.
        
        Args:
            dicom_data: DICOM data dictionary from DICOMLoader
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if dataset exists
        if "dataset" not in dicom_data:
            errors.append("Missing DICOM dataset")
            return False, errors
        
        dataset = dicom_data["dataset"]
        
        # Validate it's a pydicom Dataset
        if not isinstance(dataset, pydicom.Dataset):
            errors.append("Invalid dataset type")
            return False, errors
        
        # Check required tags
        for tag in self.required_tags:
            if not hasattr(dataset, tag) or getattr(dataset, tag) is None:
                errors.append(f"Missing required tag: {tag}")
        
        # Validate metadata if present
        if "metadata" in dicom_data:
            metadata = dicom_data["metadata"]
            
            # Check modality is valid
            modality = metadata.get("Modality")
            if modality and not self._is_valid_modality(modality):
                errors.append(f"Invalid modality: {modality}")
            
            # Check date format
            study_date = metadata.get("StudyDate")
            if study_date and not self._is_valid_date(study_date):
                errors.append(f"Invalid StudyDate format: {study_date}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.debug(f"DICOM file validated successfully: {dicom_data.get('file_name', 'unknown')}")
        else:
            logger.warning(f"DICOM validation failed with {len(errors)} errors")
        
        return is_valid, errors
    
    def validate_dataset(self, dataset: pydicom.Dataset) -> Tuple[bool, List[str]]:
        """
        Validate a pydicom Dataset directly.
        
        Args:
            dataset: pydicom Dataset object
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required tags
        for tag in self.required_tags:
            if not hasattr(dataset, tag) or getattr(dataset, tag) is None:
                errors.append(f"Missing required tag: {tag}")
        
        # Check modality
        if hasattr(dataset, "Modality"):
            if not self._is_valid_modality(dataset.Modality):
                errors.append(f"Invalid modality: {dataset.Modality}")
        
        # Check date
        if hasattr(dataset, "StudyDate"):
            if not self._is_valid_date(dataset.StudyDate):
                errors.append(f"Invalid StudyDate format: {dataset.StudyDate}")
        
        return len(errors) == 0, errors
    
    def _is_valid_modality(self, modality: str) -> bool:
        """
        Check if modality is valid.
        
        Args:
            modality: Modality string
            
        Returns:
            True if valid, False otherwise
        """
        valid_modalities = [
            "CT", "MR", "CR", "DX", "US", "NM", "PT", "XA",
            "RF", "MG", "OT", "BI", "CD", "DD", "ES", "LS",
            "ST", "RG", "TG", "RTIMAGE", "RTDOSE", "RTSTRUCT",
        ]
        return modality.upper() in valid_modalities
    
    def _is_valid_date(self, date_str: str) -> bool:
        """
        Validate DICOM date format (YYYYMMDD).
        
        Args:
            date_str: Date string
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(date_str, str):
            return False
        
        # DICOM date format is YYYYMMDD
        if len(date_str) != 8:
            return False
        
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            # Basic validation
            if year < 1900 or year > 2100:
                return False
            if month < 1 or month > 12:
                return False
            if day < 1 or day > 31:
                return False
            
            return True
        except ValueError:
            return False
