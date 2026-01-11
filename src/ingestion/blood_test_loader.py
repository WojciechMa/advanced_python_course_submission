"""Blood test data loader with unit conversion and normalization."""

from typing import Optional, List, Dict, Any, Union
import logging
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# Standard unit conversions for common lab tests
UNIT_CONVERSIONS = {
    'WBC': {  # White Blood Cell count
        'K/uL': 1.0,  # Standard: thousands per microliter
        '10^9/L': 1.0,  # Same as K/uL
        'cells/uL': 0.001,  # Convert to K/uL
    },
    'Hemoglobin': {
        'g/dL': 1.0,  # Standard: grams per deciliter
        'g/L': 0.1,  # Convert to g/dL
        'mmol/L': 1.611,  # Convert to g/dL (multiply by 1.611)
    },
    'Glucose': {
        'mg/dL': 1.0,  # Standard: milligrams per deciliter
        'mmol/L': 18.0,  # Convert to mg/dL (multiply by 18)
    },
    'Creatinine': {
        'mg/dL': 1.0,  # Standard
        'umol/L': 0.0113,  # Convert to mg/dL
    },
    'CRP': {  # C-reactive protein
        'mg/L': 1.0,  # Standard
        'mg/dL': 10.0,  # Convert to mg/L
    },
}

# Reference ranges for common tests (normal values)
REFERENCE_RANGES = {
    'WBC': {'min': 4.5, 'max': 11.0, 'unit': 'K/uL'},
    'RBC': {'min': 4.5, 'max': 5.5, 'unit': '10^6/uL'},
    'Hemoglobin': {'min': 13.5, 'max': 17.5, 'unit': 'g/dL'},  # Male
    'Hematocrit': {'min': 38.8, 'max': 50.0, 'unit': '%'},  # Male
    'Platelets': {'min': 150, 'max': 400, 'unit': 'K/uL'},
    'Glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},  # Fasting
    'BUN': {'min': 7, 'max': 20, 'unit': 'mg/dL'},
    'Creatinine': {'min': 0.7, 'max': 1.3, 'unit': 'mg/dL'},  # Male
    'Sodium': {'min': 136, 'max': 145, 'unit': 'mmol/L'},
    'Potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mmol/L'},
    'Chloride': {'min': 98, 'max': 107, 'unit': 'mmol/L'},
    'CO2': {'min': 23, 'max': 29, 'unit': 'mmol/L'},
    'Calcium': {'min': 8.5, 'max': 10.5, 'unit': 'mg/dL'},
    'CRP': {'min': 0, 'max': 3.0, 'unit': 'mg/L'},
    'Procalcitonin': {'min': 0, 'max': 0.15, 'unit': 'ng/mL'},
    'BNP': {'min': 0, 'max': 100, 'unit': 'pg/mL'},
    'Troponin': {'min': 0, 'max': 0.04, 'unit': 'ng/mL'},
    'D-dimer': {'min': 0, 'max': 0.5, 'unit': 'mg/L'},
}


class BloodTestLoader:
    """
    Load and normalize blood test data from various formats.
    
    Supports:
    - CSV files (MIMIC-IV format, custom formats)
    - JSON files
    - Pandas DataFrames
    
    Features:
    - Unit conversion to standard units
    - Reference range validation
    - Abnormal flag calculation
    - Temporal metadata extraction
    """
    
    def __init__(
        self,
        normalize_units: bool = True,
        add_reference_ranges: bool = True,
        validate_values: bool = True,
    ):
        """
        Initialize blood test loader.
        
        Args:
            normalize_units: Convert values to standard units
            add_reference_ranges: Add reference range columns
            validate_values: Check for abnormal values
        """
        self.normalize_units = normalize_units
        self.add_reference_ranges = add_reference_ranges
        self.validate_values = validate_values
        
        self._stats = {
            'total_records': 0,
            'successfully_loaded': 0,
            'conversion_errors': 0,
            'validation_errors': 0,
        }
        
        logger.info(
            f"Initialized BloodTestLoader: normalize={normalize_units}, "
            f"validate={validate_values}"
        )
    
    def load_csv(
        self,
        file_path: str,
        format_type: str = 'mimic',
    ) -> pd.DataFrame:
        """
        Load blood test data from CSV file.
        
        Args:
            file_path: Path to CSV file
            format_type: Format of CSV ('mimic', 'custom')
            
        Returns:
            DataFrame with blood test data
        """
        logger.info(f"Loading blood test data from CSV: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self._stats['total_records'] = len(df)
            
            # Process based on format type
            if format_type == 'mimic':
                df = self._process_mimic_format(df)
            else:
                df = self._process_custom_format(df)
            
            # Apply transformations
            if self.normalize_units:
                df = self.normalize_values(df)
            
            if self.add_reference_ranges:
                df = self._add_reference_ranges(df)
            
            if self.validate_values:
                df = self._validate_values(df)
            
            self._stats['successfully_loaded'] = len(df)
            logger.info(f"Successfully loaded {len(df)} lab test records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def load_json(self, file_path: str) -> pd.DataFrame:
        """
        Load blood test data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            DataFrame with blood test data
        """
        logger.info(f"Loading blood test data from JSON: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("JSON must contain list or dict")
            
            self._stats['total_records'] = len(df)
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Apply transformations
            if self.normalize_units:
                df = self.normalize_values(df)
            
            if self.add_reference_ranges:
                df = self._add_reference_ranges(df)
            
            if self.validate_values:
                df = self._validate_values(df)
            
            self._stats['successfully_loaded'] = len(df)
            logger.info(f"Successfully loaded {len(df)} lab test records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise
    
    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process an existing DataFrame with blood test data.
        
        Args:
            df: DataFrame with blood test data
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing DataFrame with {len(df)} records")
        
        self._stats['total_records'] = len(df)
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Apply transformations
        if self.normalize_units:
            df = self.normalize_values(df)
        
        if self.add_reference_ranges:
            df = self._add_reference_ranges(df)
        
        if self.validate_values:
            df = self._validate_values(df)
        
        self._stats['successfully_loaded'] = len(df)
        
        return df
    
    def normalize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize blood test values to standard units.
        
        Args:
            df: DataFrame with raw blood test data
            
        Returns:
            DataFrame with normalized values
        """
        if 'lab_name' not in df.columns or 'value' not in df.columns:
            logger.warning("Missing required columns for normalization")
            return df
        
        df = df.copy()
        
        # Ensure value is numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Apply unit conversions if unit column exists
        if 'unit' in df.columns:
            for idx, row in df.iterrows():
                lab_name = row['lab_name']
                unit = row['unit']
                value = row['value']
                
                if pd.isna(value):
                    continue
                
                # Check if conversion is needed
                if lab_name in UNIT_CONVERSIONS:
                    if unit in UNIT_CONVERSIONS[lab_name]:
                        conversion_factor = UNIT_CONVERSIONS[lab_name][unit]
                        df.at[idx, 'value'] = value * conversion_factor
                        
                        # Update unit to standard
                        standard_unit = list(UNIT_CONVERSIONS[lab_name].keys())[0]
                        df.at[idx, 'unit'] = standard_unit
                        
                        logger.debug(
                            f"Converted {lab_name}: {value} {unit} -> "
                            f"{df.at[idx, 'value']:.2f} {standard_unit}"
                        )
        
        return df
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get loading statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self._stats.copy()
    
    # Private helper methods
    
    def _process_mimic_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process MIMIC-IV labevents format.
        
        Expected columns:
        - subject_id: Patient ID
        - hadm_id: Hospital admission ID
        - charttime: Lab test time
        - itemid: Lab test item ID
        - value: Lab value
        - valueuom: Unit of measurement
        """
        # Standardize column names
        column_mapping = {
            'itemid': 'lab_id',
            'valueuom': 'unit',
            'charttime': 'test_datetime',
        }
        df = df.rename(columns=column_mapping)
        
        # Parse datetime
        if 'test_datetime' in df.columns:
            df['test_datetime'] = pd.to_datetime(df['test_datetime'], errors='coerce')
        
        # Add lab_name if not present (would need lab item dictionary in real use)
        if 'lab_name' not in df.columns and 'lab_id' in df.columns:
            df['lab_name'] = 'LAB_' + df['lab_id'].astype(str)
        
        return df
    
    def _process_custom_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process custom CSV format."""
        df = self._standardize_columns(df)
        
        # Parse datetime if present
        datetime_cols = ['test_datetime', 'test_date', 'datetime', 'charttime']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if col != 'test_datetime':
                    df = df.rename(columns={col: 'test_datetime'})
                break
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format."""
        # Common column name mappings
        mappings = {
            'patient_id': 'subject_id',
            'patientid': 'subject_id',
            'test_name': 'lab_name',
            'testname': 'lab_name',
            'result': 'value',
            'test_result': 'value',
            'units': 'unit',
            'test_date': 'test_datetime',
            'date': 'test_datetime',
        }
        
        # Apply mappings
        df = df.rename(columns=mappings)
        
        return df
    
    def _add_reference_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add reference range columns."""
        if 'lab_name' not in df.columns:
            return df
        
        df = df.copy()
        df['ref_min'] = None
        df['ref_max'] = None
        df['ref_unit'] = None
        
        for idx, row in df.iterrows():
            lab_name = row['lab_name']
            if lab_name in REFERENCE_RANGES:
                ref_range = REFERENCE_RANGES[lab_name]
                df.at[idx, 'ref_min'] = ref_range['min']
                df.at[idx, 'ref_max'] = ref_range['max']
                df.at[idx, 'ref_unit'] = ref_range['unit']
        
        return df
    
    def _validate_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate lab values and add abnormal flags."""
        if 'value' not in df.columns or 'lab_name' not in df.columns:
            return df
        
        df = df.copy()
        df['is_abnormal'] = False
        df['abnormal_flag'] = 'NORMAL'
        
        for idx, row in df.iterrows():
            value = row['value']
            lab_name = row['lab_name']
            
            if pd.isna(value):
                df.at[idx, 'abnormal_flag'] = 'MISSING'
                continue
            
            # Check against reference ranges
            if lab_name in REFERENCE_RANGES:
                ref_range = REFERENCE_RANGES[lab_name]
                
                if value < ref_range['min']:
                    df.at[idx, 'is_abnormal'] = True
                    df.at[idx, 'abnormal_flag'] = 'LOW'
                elif value > ref_range['max']:
                    df.at[idx, 'is_abnormal'] = True
                    df.at[idx, 'abnormal_flag'] = 'HIGH'
            
            # Check for physiologically impossible values
            if value < 0 and lab_name not in ['Temperature']:  # Most labs can't be negative
                df.at[idx, 'abnormal_flag'] = 'INVALID'
                self._stats['validation_errors'] += 1
                logger.warning(f"Invalid value for {lab_name}: {value}")
        
        return df
