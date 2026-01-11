"""Anonymize patient data to ensure HIPAA compliance."""

import logging
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Union
import pydicom
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Anonymizer:
    """Anonymize Protected Health Information (PHI) from medical data."""
    
    # DICOM tags that contain PHI and must be removed/anonymized
    # Note: PatientID is handled separately (hashed, not removed)
    PHI_TAGS = [
        # Patient identification
        "PatientName",
        "PatientBirthDate",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "PatientMotherBirthName",
        # Institution and personnel
        "InstitutionName",
        "InstitutionAddress",
        "InstitutionalDepartmentName",
        "ReferringPhysicianName",
        "ReferringPhysicianAddress",
        "ReferringPhysicianTelephoneNumbers",
        "PerformingPhysicianName",
        "OperatorsName",
        "PhysiciansOfRecord",
        # Study related
        "StudyID",
        "AccessionNumber",
        # Other identifiable information
        "OtherPatientIDs",
        "OtherPatientNames",
        "PatientInsurancePlanCodeSequence",
        "PatientReligiousPreference",
        "EthnicGroup",
        "Occupation",
        "MilitaryRank",
        "BranchOfService",
        "CountryOfResidence",
        "RegionOfResidence",
        "PatientComments",
        "RequestingPhysician",
        "RequestingService",
    ]
    
    def __init__(
        self,
        anonymization_level: str = "strict",
        salt: Optional[str] = None,
        date_shift_days: int = 0,
    ):
        """
        Initialize anonymizer.
        
        Args:
            anonymization_level: Level of anonymization
                - 'strict': Remove all PHI
                - 'moderate': Keep some non-identifying info
                - 'none': No anonymization (for testing only)
            salt: Salt for hashing patient IDs (generated if not provided)
            date_shift_days: Number of days to shift dates (for temporal analysis)
        """
        self.anonymization_level = anonymization_level
        self.salt = salt or secrets.token_hex(32)
        self.date_shift_days = date_shift_days
        
        # Mapping of original IDs to anonymized IDs for consistency
        self._id_mapping: Dict[str, str] = {}
        
        logger.info(
            f"Initialized Anonymizer: level={anonymization_level}, "
            f"date_shift={date_shift_days} days"
        )
        
        if anonymization_level == "none":
            logger.warning("⚠️  Anonymization is DISABLED - Do not use with real patient data!")
    
    def anonymize_dicom(
        self,
        dicom_dataset: pydicom.Dataset,
        preserve_metadata: bool = True,
    ) -> pydicom.Dataset:
        """
        Remove or hash PHI from DICOM dataset.
        
        Args:
            dicom_dataset: DICOM dataset to anonymize
            preserve_metadata: Whether to keep non-PHI metadata
            
        Returns:
            Anonymized DICOM dataset
        """
        if self.anonymization_level == "none":
            return dicom_dataset
        
        # Create a copy to avoid modifying original
        anon_dataset = dicom_dataset.copy()
        
        # Anonymize patient ID
        if hasattr(anon_dataset, "PatientID"):
            original_id = str(anon_dataset.PatientID)
            anon_dataset.PatientID = self.hash_patient_id(original_id)
        
        # Remove or anonymize PHI tags
        for tag in self.PHI_TAGS:
            if hasattr(anon_dataset, tag):
                if self.anonymization_level == "strict":
                    # Remove tag completely
                    delattr(anon_dataset, tag)
                elif self.anonymization_level == "moderate":
                    # Replace with placeholder
                    if tag in ["PatientName", "ReferringPhysicianName", "PerformingPhysicianName"]:
                        setattr(anon_dataset, tag, "ANONYMIZED")
                    else:
                        delattr(anon_dataset, tag)
        
        # Anonymize dates if date shifting is enabled
        if self.date_shift_days != 0:
            anon_dataset = self._shift_dates(anon_dataset)
        
        # Add anonymization marker
        anon_dataset.PatientIdentityRemoved = "YES"
        anon_dataset.DeidentificationMethod = f"Anonymizer v1.0 - {self.anonymization_level}"
        
        logger.debug(f"Anonymized DICOM dataset with {len(self.PHI_TAGS)} PHI tags checked")
        
        return anon_dataset
    
    def anonymize_dicom_dict(self, dicom_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize DICOM data dictionary (from DICOMLoader).
        
        Args:
            dicom_data: DICOM data dictionary containing 'dataset' and 'metadata'
            
        Returns:
            Anonymized DICOM data dictionary
        """
        if self.anonymization_level == "none":
            return dicom_data
        
        anon_data = dicom_data.copy()
        
        # Anonymize the dataset
        if "dataset" in anon_data:
            anon_data["dataset"] = self.anonymize_dicom(anon_data["dataset"])
        
        # Anonymize metadata
        if "metadata" in anon_data:
            anon_data["metadata"] = self._anonymize_metadata(anon_data["metadata"])
        
        # Remove file path information in strict mode
        if self.anonymization_level == "strict":
            if "file_path" in anon_data:
                del anon_data["file_path"]
        
        return anon_data
    
    def anonymize_blood_test(self, blood_test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize blood test data.
        
        Args:
            blood_test_data: Blood test data dictionary
            
        Returns:
            Anonymized blood test data
        """
        if self.anonymization_level == "none":
            return blood_test_data
        
        anon_data = blood_test_data.copy()
        
        # Anonymize patient ID
        if "patient_id" in anon_data:
            original_id = str(anon_data["patient_id"])
            anon_data["patient_id"] = self.hash_patient_id(original_id)
        
        # Shift dates if configured
        if self.date_shift_days != 0 and "test_date" in anon_data:
            anon_data["test_date"] = self._shift_date_string(anon_data["test_date"])
        
        # Remove PHI fields in strict mode
        if self.anonymization_level == "strict":
            phi_fields = ["patient_name", "mrn", "ssn", "address", "phone"]
            for field in phi_fields:
                if field in anon_data:
                    del anon_data[field]
        
        return anon_data
    
    def hash_patient_id(self, patient_id: str) -> str:
        """
        Generate anonymized patient ID using hashing.
        
        Consistent hashing ensures the same patient_id always produces
        the same anonymized ID, allowing data linkage while protecting identity.
        
        Args:
            patient_id: Original patient ID
            
        Returns:
            Hashed patient ID (16 character hex string)
        """
        # Check if we've already mapped this ID
        if patient_id in self._id_mapping:
            return self._id_mapping[patient_id]
        
        # Create hash with salt
        combined = f"{patient_id}{self.salt}"
        hash_obj = hashlib.sha256(combined.encode())
        anonymized_id = f"ANON_{hash_obj.hexdigest()[:12].upper()}"
        
        # Store mapping for consistency
        self._id_mapping[patient_id] = anonymized_id
        
        logger.debug(f"Hashed patient ID: {patient_id[:4]}... -> {anonymized_id}")
        
        return anonymized_id
    
    def _anonymize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize metadata dictionary.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Anonymized metadata
        """
        anon_metadata = metadata.copy()
        
        # Anonymize patient ID
        if "PatientID" in anon_metadata:
            original_id = str(anon_metadata["PatientID"])
            anon_metadata["PatientID"] = self.hash_patient_id(original_id)
        
        # Remove PHI from metadata
        for tag in self.PHI_TAGS:
            if tag in anon_metadata:
                if self.anonymization_level == "strict":
                    del anon_metadata[tag]
                elif self.anonymization_level == "moderate":
                    if tag in ["PatientName", "ReferringPhysicianName"]:
                        anon_metadata[tag] = "ANONYMIZED"
                    else:
                        del anon_metadata[tag]
        
        return anon_metadata
    
    def _shift_dates(self, dataset: pydicom.Dataset) -> pydicom.Dataset:
        """
        Shift dates in DICOM dataset by configured number of days.
        
        This preserves temporal relationships while obscuring actual dates.
        
        Args:
            dataset: DICOM dataset
            
        Returns:
            Dataset with shifted dates
        """
        date_tags = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "PatientBirthDate"]
        
        for tag in date_tags:
            if hasattr(dataset, tag):
                try:
                    original_date = getattr(dataset, tag)
                    if original_date and len(original_date) == 8:
                        # Parse DICOM date format (YYYYMMDD)
                        date_obj = datetime.strptime(original_date, "%Y%m%d")
                        # Shift by configured days
                        shifted_date = date_obj + timedelta(days=self.date_shift_days)
                        # Convert back to DICOM format
                        setattr(dataset, tag, shifted_date.strftime("%Y%m%d"))
                except Exception as e:
                    logger.warning(f"Could not shift date for tag {tag}: {e}")
        
        return dataset
    
    def _shift_date_string(self, date_string: str) -> str:
        """
        Shift a date string by configured number of days.
        
        Args:
            date_string: Date string in ISO format (YYYY-MM-DD) or DICOM format (YYYYMMDD)
            
        Returns:
            Shifted date string in same format
        """
        try:
            # Try ISO format first
            if "-" in date_string:
                date_obj = datetime.fromisoformat(date_string.split("T")[0])
                shifted_date = date_obj + timedelta(days=self.date_shift_days)
                return shifted_date.date().isoformat()
            # Try DICOM format
            elif len(date_string) == 8:
                date_obj = datetime.strptime(date_string, "%Y%m%d")
                shifted_date = date_obj + timedelta(days=self.date_shift_days)
                return shifted_date.strftime("%Y%m%d")
            else:
                return date_string
        except Exception as e:
            logger.warning(f"Could not shift date string {date_string}: {e}")
            return date_string
    
    def get_id_mapping(self) -> Dict[str, str]:
        """
        Get the mapping of original to anonymized IDs.
        
        ⚠️  WARNING: This contains PHI! Store securely and separately from anonymized data.
        
        Returns:
            Dictionary mapping original IDs to anonymized IDs
        """
        logger.warning("⚠️  ID mapping contains PHI - handle with care!")
        return self._id_mapping.copy()
    
    def clear_id_mapping(self) -> None:
        """Clear the ID mapping cache."""
        self._id_mapping.clear()
        logger.info("Cleared ID mapping cache")
    
    def export_id_mapping(self, filepath: str, encrypt: bool = True) -> None:
        """
        Export ID mapping to file.
        
        ⚠️  WARNING: This file contains PHI!
        
        Args:
            filepath: Path to save mapping
            encrypt: Whether to encrypt the file (recommended)
        """
        logger.warning(f"⚠️  Exporting ID mapping (contains PHI) to: {filepath}")
        
        import json
        
        if encrypt:
            # TODO: Implement encryption
            logger.warning("Encryption not yet implemented - saving as plaintext")
        
        with open(filepath, "w") as f:
            json.dump({
                "anonymization_level": self.anonymization_level,
                "date_shift_days": self.date_shift_days,
                "timestamp": datetime.now().isoformat(),
                "mappings": self._id_mapping,
            }, f, indent=2)
        
        logger.info(f"ID mapping exported with {len(self._id_mapping)} entries")
