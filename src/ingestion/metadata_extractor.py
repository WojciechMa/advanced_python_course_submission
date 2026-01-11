"""Extract metadata from DICOM files."""

import logging
from typing import Dict, Any, Optional, List
import pydicom
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract relevant DICOM tags and metadata."""
    
    def __init__(self, custom_tags: Optional[List[str]] = None):
        """
        Initialize metadata extractor.
        
        Args:
            custom_tags: Additional DICOM tags to extract beyond defaults
        """
        self.required_tags = [
            "PatientID",
            "StudyDate",
            "Modality",
            "BodyPartExamined",
        ]
        
        self.optional_tags = [
            "PatientName",
            "PatientBirthDate",
            "PatientSex",
            "StudyTime",
            "StudyDescription",
            "SeriesDescription",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "Manufacturer",
            "ManufacturerModelName",
            "Rows",
            "Columns",
            "PixelSpacing",
            "SliceThickness",
            "WindowCenter",
            "WindowWidth",
            "InstitutionName",
            "ReferringPhysicianName",
            "PerformingPhysicianName",
        ]
        
        if custom_tags:
            self.optional_tags.extend(custom_tags)
        
        logger.info("Initialized MetadataExtractor")
    
    def extract(self, dicom_dataset: pydicom.Dataset) -> Dict[str, Any]:
        """
        Extract metadata from DICOM file.
        
        Args:
            dicom_dataset: DICOM dataset object
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            "extraction_timestamp": datetime.now().isoformat(),
            "sop_class_uid": None,
        }
        
        # Extract SOP Class UID
        if hasattr(dicom_dataset, "SOPClassUID"):
            metadata["sop_class_uid"] = str(dicom_dataset.SOPClassUID)
        
        # Extract required tags
        for tag in self.required_tags:
            metadata[tag] = self._get_tag_value(dicom_dataset, tag)
        
        # Extract optional tags
        for tag in self.optional_tags:
            value = self._get_tag_value(dicom_dataset, tag)
            if value is not None:
                metadata[tag] = value
        
        # Extract image-specific metadata
        metadata.update(self._extract_image_metadata(dicom_dataset))
        
        # Parse dates and times
        metadata.update(self._parse_temporal_data(metadata))
        
        return metadata
    
    def extract_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata directly from DICOM file path.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Dictionary of extracted metadata or None if failed
        """
        try:
            dataset = pydicom.dcmread(file_path, stop_before_pixels=True)
            metadata = self.extract(dataset)
            metadata["file_path"] = file_path
            return metadata
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return None
    
    def _get_tag_value(self, dataset: pydicom.Dataset, tag_name: str) -> Any:
        """
        Safely get a tag value from dataset.
        
        Args:
            dataset: DICOM dataset
            tag_name: Name of the tag
            
        Returns:
            Tag value or None if not present
        """
        try:
            if not hasattr(dataset, tag_name):
                return None
            
            value = getattr(dataset, tag_name)
            
            if value is None:
                return None
            
            # Handle different value types
            if isinstance(value, pydicom.multival.MultiValue):
                return list(value)
            elif isinstance(value, bytes):
                return value.decode("utf-8", errors="ignore").strip()
            elif isinstance(value, str):
                return value.strip()
            else:
                return str(value)
                
        except Exception as e:
            logger.debug(f"Could not extract {tag_name}: {e}")
            return None
    
    def _extract_image_metadata(self, dataset: pydicom.Dataset) -> Dict[str, Any]:
        """
        Extract image-specific metadata.
        
        Args:
            dataset: DICOM dataset
            
        Returns:
            Dictionary with image metadata
        """
        image_meta = {}
        
        # Image dimensions
        if hasattr(dataset, "Rows") and hasattr(dataset, "Columns"):
            image_meta["image_shape"] = [int(dataset.Rows), int(dataset.Columns)]
        
        # Pixel spacing
        if hasattr(dataset, "PixelSpacing"):
            try:
                image_meta["pixel_spacing_mm"] = [float(x) for x in dataset.PixelSpacing]
            except:
                pass
        
        # Slice thickness
        if hasattr(dataset, "SliceThickness"):
            try:
                image_meta["slice_thickness_mm"] = float(dataset.SliceThickness)
            except:
                pass
        
        # Window settings
        if hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth"):
            try:
                wc = dataset.WindowCenter
                ww = dataset.WindowWidth
                # Handle multiple window settings
                if isinstance(wc, pydicom.multival.MultiValue):
                    wc = float(wc[0])
                    ww = float(ww[0])
                else:
                    wc = float(wc)
                    ww = float(ww)
                image_meta["window_center"] = wc
                image_meta["window_width"] = ww
            except:
                pass
        
        # Bits allocated/stored
        for tag in ["BitsAllocated", "BitsStored", "HighBit"]:
            if hasattr(dataset, tag):
                try:
                    image_meta[tag] = int(getattr(dataset, tag))
                except:
                    pass
        
        return image_meta
    
    def _parse_temporal_data(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse date and time fields into structured format.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with parsed temporal data
        """
        temporal = {}
        
        # Parse study date
        study_date = metadata.get("StudyDate")
        if study_date and len(study_date) == 8:
            try:
                temporal["study_date_parsed"] = datetime.strptime(study_date, "%Y%m%d").date().isoformat()
            except:
                pass
        
        # Parse study time
        study_time = metadata.get("StudyTime")
        if study_time:
            try:
                # DICOM time format can be HHMMSS.FFFFFF
                time_parts = study_time.split(".")[0]  # Remove fractional seconds
                if len(time_parts) >= 6:
                    temporal["study_time_parsed"] = f"{time_parts[:2]}:{time_parts[2:4]}:{time_parts[4:6]}"
            except:
                pass
        
        # Combine date and time if both available
        if "study_date_parsed" in temporal and "study_time_parsed" in temporal:
            try:
                dt_str = f"{temporal['study_date_parsed']}T{temporal['study_time_parsed']}"
                temporal["study_datetime"] = dt_str
            except:
                pass
        
        # Parse patient birth date
        birth_date = metadata.get("PatientBirthDate")
        if birth_date and len(birth_date) == 8:
            try:
                temporal["patient_birth_date_parsed"] = datetime.strptime(birth_date, "%Y%m%d").date().isoformat()
            except:
                pass
        
        return temporal
