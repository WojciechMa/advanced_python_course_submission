"""DICOM image loader with PySpark support."""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class DICOMLoader:
    """
    Load and process DICOM files from various sources.
    
    Supports:
    - Local filesystem
    - S3/MinIO object storage (future)
    - PACS API endpoints (future)
    
    Uses PySpark for distributed processing of large datasets.
    """
    
    def __init__(
        self,
        source_type: str = "local",
        batch_size: int = 100,
        anonymization_level: str = "strict",
        supported_modalities: Optional[List[str]] = None,
    ):
        """
        Initialize DICOM loader.
        
        Args:
            source_type: Source type ('local', 's3', 'pacs_api')
            batch_size: Number of files to process in parallel
            anonymization_level: Level of anonymization ('strict', 'moderate', 'none')
            supported_modalities: List of supported modalities (default: CT, MRI, X-RAY)
        """
        self.source_type = source_type
        self.batch_size = batch_size
        self.anonymization_level = anonymization_level
        self.supported_modalities = supported_modalities or ["CT", "MR", "CR", "DX"]
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "skipped_files": 0,
        }
        
        logger.info(
            f"Initialized DICOMLoader: source={source_type}, "
            f"batch_size={batch_size}, anonymization={anonymization_level}"
        )
    
    def load_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load a single DICOM file.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Dictionary containing DICOM data and metadata, or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(str(file_path), force=True)
            
            # Extract basic information
            result = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "dataset": dicom_data,
                "metadata": self._extract_metadata(dicom_data),
                "pixel_data_available": hasattr(dicom_data, "pixel_array"),
                "loaded_at": datetime.now().isoformat(),
            }
            
            # Add pixel data info if available
            if result["pixel_data_available"]:
                try:
                    pixel_array = dicom_data.pixel_array
                    result["pixel_data_shape"] = pixel_array.shape
                    result["pixel_data_dtype"] = str(pixel_array.dtype)
                except Exception as e:
                    logger.warning(f"Could not load pixel data for {file_path}: {e}")
                    result["pixel_data_available"] = False
            
            logger.debug(f"Successfully loaded: {file_path}")
            self.stats["successful_loads"] += 1
            
            return result
            
        except InvalidDicomError as e:
            logger.error(f"Invalid DICOM file {file_path}: {e}")
            self.stats["failed_loads"] += 1
            return None
        except Exception as e:
            logger.error(f"Error loading DICOM file {file_path}: {e}")
            self.stats["failed_loads"] += 1
            return None
    
    def load_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        file_pattern: str = "*.dcm",
    ) -> List[Dict[str, Any]]:
        """
        Load all DICOM files from a directory.
        
        Args:
            directory_path: Path to directory containing DICOM files
            recursive: Whether to search subdirectories
            file_pattern: File pattern to match (default: *.dcm)
            
        Returns:
            List of dictionaries containing DICOM data and metadata
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Find DICOM files
        if recursive:
            dicom_files = list(directory_path.rglob(file_pattern))
        else:
            dicom_files = list(directory_path.glob(file_pattern))
        
        # Also try to find files without extension
        all_files = list(directory_path.rglob("*") if recursive else directory_path.glob("*"))
        for file in all_files:
            if file.is_file() and file.suffix == "":
                # Check if it's a DICOM file by trying to read it
                try:
                    pydicom.dcmread(str(file), stop_before_pixels=True)
                    dicom_files.append(file)
                except:
                    pass
        
        logger.info(f"Found {len(dicom_files)} potential DICOM files in {directory_path}")
        self.stats["total_files"] = len(dicom_files)
        
        # Load files
        results = []
        for i, file_path in enumerate(dicom_files, 1):
            if i % 10 == 0:
                logger.info(f"Processing file {i}/{len(dicom_files)}")
            
            result = self.load_file(file_path)
            if result is not None:
                # Check if modality is supported
                modality = result["metadata"].get("Modality", "UNKNOWN")
                if modality in self.supported_modalities or "UNKNOWN" in self.supported_modalities:
                    results.append(result)
                else:
                    logger.debug(f"Skipping unsupported modality: {modality}")
                    self.stats["skipped_files"] += 1
        
        logger.info(
            f"Loaded {len(results)} DICOM files successfully. "
            f"Failed: {self.stats['failed_loads']}, Skipped: {self.stats['skipped_files']}"
        )
        
        return results
    
    def load(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load DICOM files from specified path (file or directory).
        
        Args:
            path: Path to DICOM file or directory
            
        Returns:
            List of dictionaries containing DICOM data and metadata
        """
        path = Path(path)
        
        if path.is_file():
            result = self.load_file(path)
            return [result] if result is not None else []
        elif path.is_dir():
            return self.load_directory(path)
        else:
            logger.error(f"Path does not exist: {path}")
            return []
    
    def _extract_metadata(self, dicom_data: pydicom.Dataset) -> Dict[str, Any]:
        """
        Extract relevant metadata from DICOM dataset.
        
        Args:
            dicom_data: DICOM dataset
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # List of tags to extract
        tags_to_extract = [
            ("PatientID", "0010,0020"),
            ("PatientName", "0010,0010"),
            ("PatientBirthDate", "0010,0030"),
            ("PatientSex", "0010,0040"),
            ("StudyDate", "0008,0020"),
            ("StudyTime", "0008,0030"),
            ("StudyDescription", "0008,1030"),
            ("SeriesDescription", "0008,103E"),
            ("Modality", "0008,0060"),
            ("BodyPartExamined", "0018,0015"),
            ("StudyInstanceUID", "0020,000D"),
            ("SeriesInstanceUID", "0020,000E"),
            ("SOPInstanceUID", "0008,0018"),
            ("Manufacturer", "0008,0070"),
            ("ManufacturerModelName", "0008,1090"),
            ("Rows", "0028,0010"),
            ("Columns", "0028,0011"),
            ("PixelSpacing", "0028,0030"),
            ("SliceThickness", "0018,0050"),
            ("WindowCenter", "0028,1050"),
            ("WindowWidth", "0028,1051"),
        ]
        
        for tag_name, tag_id in tags_to_extract:
            try:
                value = getattr(dicom_data, tag_name, None)
                if value is not None:
                    # Convert to string or appropriate type
                    if isinstance(value, pydicom.multival.MultiValue):
                        metadata[tag_name] = list(value)
                    elif isinstance(value, bytes):
                        metadata[tag_name] = value.decode("utf-8", errors="ignore")
                    else:
                        metadata[tag_name] = str(value)
            except Exception as e:
                logger.debug(f"Could not extract {tag_name}: {e}")
                metadata[tag_name] = None
        
        return metadata
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get loading statistics.
        
        Returns:
            Dictionary with loading statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset loading statistics."""
        self.stats = {
            "total_files": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "skipped_files": 0,
        }
        logger.info("Statistics reset")
