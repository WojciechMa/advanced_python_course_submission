"""Image preprocessing for medical imaging data.

This module provides comprehensive image preprocessing functionality including:
- DICOM to numpy array conversion with windowing
- Resizing and normalization
- Data augmentation for training
- Train/validation/test splitting
- Batch processing for model input
"""

import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
import logging
from sklearn.model_selection import train_test_split
from PIL import Image
import json

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess medical images for deep learning models.
    
    Features:
    - DICOM loading with customizable windowing
    - Resize to target dimensions
    - Normalize pixel values to [0, 1] or [-1, 1]
    - Data augmentation (rotation, flip, brightness, contrast)
    - Dataset splitting with stratification
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        window_center: Optional[int] = None,
        window_width: Optional[int] = None,
        normalize_method: str = 'zero_one',
        augmentation: bool = False,
        random_seed: int = 42
    ):
        """Initialize image preprocessor.
        
        Args:
            target_size: Target image dimensions (height, width)
            window_center: DICOM windowing center (e.g., -600 for lung)
            window_width: DICOM windowing width (e.g., 1500 for lung)
            normalize_method: 'zero_one' for [0,1], 'neg_one_one' for [-1,1], 
                            'standardize' for zero mean unit variance
            augmentation: Enable data augmentation
            random_seed: Random seed for reproducibility
        """
        self.target_size = target_size
        self.window_center = window_center
        self.window_width = window_width
        self.normalize_method = normalize_method
        self.augmentation = augmentation
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Statistics for normalization
        self.mean = None
        self.std = None
        
        logger.info(f"ImagePreprocessor initialized: size={target_size}, "
                   f"normalize={normalize_method}, augment={augmentation}")
    
    def load_dicom_image(
        self, 
        dicom_path: Union[str, Path]
    ) -> np.ndarray:
        """Load and extract pixel array from DICOM file.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            2D numpy array with pixel data
        """
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Apply rescale slope and intercept if present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        return pixel_array
    
    def apply_windowing(
        self, 
        pixel_array: np.ndarray,
        center: Optional[int] = None,
        width: Optional[int] = None
    ) -> np.ndarray:
        """Apply DICOM windowing to pixel array.
        
        Args:
            pixel_array: Input pixel array
            center: Window center (uses instance default if None)
            width: Window width (uses instance default if None)
            
        Returns:
            Windowed pixel array in range [0, 255]
        """
        center = center if center is not None else self.window_center
        width = width if width is not None else self.window_width
        
        if center is None or width is None:
            # No windowing, just scale to 0-255
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            if pixel_max > pixel_min:
                return ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
            return pixel_array.astype(np.uint8)
        
        # Apply window level/width
        lower = center - width / 2
        upper = center + width / 2
        
        windowed = np.clip(pixel_array, lower, upper)
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        
        return windowed
    
    def resize_image(
        self, 
        image: np.ndarray,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Resize image to target dimensions.
        
        Args:
            image: Input image array
            size: Target size (uses instance default if None)
            
        Returns:
            Resized image array
        """
        size = size if size is not None else self.target_size
        
        # Use PIL for high-quality resizing
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        resized = pil_image.resize((size[1], size[0]), Image.LANCZOS)
        
        return np.array(resized, dtype=np.float32)
    
    def normalize_image(
        self, 
        image: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """Normalize image pixel values.
        
        Args:
            image: Input image array
            method: Normalization method (uses instance default if None)
            
        Returns:
            Normalized image array
        """
        method = method if method is not None else self.normalize_method
        
        if method == 'zero_one':
            # Scale to [0, 1]
            image_min = image.min()
            image_max = image.max()
            if image_max > image_min:
                return (image - image_min) / (image_max - image_min)
            return image / 255.0 if image.max() > 1 else image
        
        elif method == 'neg_one_one':
            # Scale to [-1, 1]
            image_min = image.min()
            image_max = image.max()
            if image_max > image_min:
                normalized = (image - image_min) / (image_max - image_min)
                return normalized * 2 - 1
            return image / 127.5 - 1
        
        elif method == 'standardize':
            # Zero mean, unit variance
            if self.mean is None or self.std is None:
                self.mean = image.mean()
                self.std = image.std()
            return (image - self.mean) / (self.std + 1e-7)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def augment_image(
        self, 
        image: np.ndarray,
        rotation_range: float = 15.0,
        flip_horizontal: bool = True,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """Apply random augmentations to image.
        
        Args:
            image: Input image array (normalized)
            rotation_range: Max rotation angle in degrees
            flip_horizontal: Enable horizontal flipping
            brightness_range: Brightness adjustment range (min, max)
            contrast_range: Contrast adjustment range (min, max)
            
        Returns:
            Augmented image array
        """
        if not self.augmentation:
            return image
        
        # Convert to PIL for augmentation
        if image.min() < 0:
            # [-1, 1] range
            pil_image = Image.fromarray(((image + 1) * 127.5).astype(np.uint8))
        else:
            # [0, 1] range
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Random rotation
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            pil_image = pil_image.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
        
        # Random horizontal flip
        if flip_horizontal and np.random.random() > 0.5:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert back to numpy
        augmented = np.array(pil_image, dtype=np.float32)
        
        # Random brightness
        brightness_factor = np.random.uniform(*brightness_range)
        augmented = augmented * brightness_factor
        
        # Random contrast
        contrast_factor = np.random.uniform(*contrast_range)
        mean_value = augmented.mean()
        augmented = (augmented - mean_value) * contrast_factor + mean_value
        
        # Clip and renormalize
        augmented = np.clip(augmented, 0, 255)
        
        # Return in original normalization range
        if image.min() < 0:
            return augmented / 127.5 - 1
        else:
            return augmented / 255.0
    
    def preprocess_image(
        self,
        dicom_path: Union[str, Path],
        augment: bool = False
    ) -> np.ndarray:
        """Complete preprocessing pipeline for a single image.
        
        Args:
            dicom_path: Path to DICOM file
            augment: Apply augmentation (overrides instance setting)
            
        Returns:
            Preprocessed image array ready for model input
        """
        # Load DICOM
        pixel_array = self.load_dicom_image(dicom_path)
        
        # Apply windowing
        windowed = self.apply_windowing(pixel_array)
        
        # Resize
        resized = self.resize_image(windowed)
        
        # Normalize
        normalized = self.normalize_image(resized)
        
        # Augment if requested
        if augment:
            normalized = self.augment_image(normalized)
        
        return normalized
    
    def preprocess_batch(
        self,
        dicom_paths: List[Union[str, Path]],
        augment: bool = False
    ) -> np.ndarray:
        """Preprocess multiple images into a batch.
        
        Args:
            dicom_paths: List of DICOM file paths
            augment: Apply augmentation to all images
            
        Returns:
            4D array (batch_size, height, width, channels)
        """
        images = []
        for path in dicom_paths:
            img = self.preprocess_image(path, augment=augment)
            images.append(img)
        
        # Stack into batch (add channel dimension)
        batch = np.array(images)
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=-1)  # Add channel dimension
        
        return batch
    
    def create_dataset_split(
        self,
        dicom_paths: List[Union[str, Path]],
        labels: List[int],
        metadata: Optional[List[Dict]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True
    ) -> Dict[str, Dict[str, Union[List, np.ndarray]]]:
        """Split dataset into train/validation/test sets.
        
        Args:
            dicom_paths: List of DICOM file paths
            labels: List of integer labels
            metadata: Optional metadata for each image
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify: Maintain class distribution in splits
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing
            'paths', 'labels', and optionally 'metadata'
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Convert to numpy arrays
        paths_array = np.array(dicom_paths)
        labels_array = np.array(labels)
        
        stratify_by = labels_array if stratify else None
        
        # First split: train vs (val+test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths_array, labels_array,
            test_size=(val_ratio + test_ratio),
            random_state=self.random_seed,
            stratify=stratify_by
        )
        
        # Second split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        stratify_temp = temp_labels if stratify else None
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=val_test_ratio,
            random_state=self.random_seed,
            stratify=stratify_temp
        )
        
        # Create result dictionary
        result = {
            'train': {
                'paths': train_paths.tolist(),
                'labels': train_labels.tolist()
            },
            'val': {
                'paths': val_paths.tolist(),
                'labels': val_labels.tolist()
            },
            'test': {
                'paths': test_paths.tolist(),
                'labels': test_labels.tolist()
            }
        }
        
        # Add metadata if provided
        if metadata is not None:
            metadata_array = np.array(metadata)
            train_idx = [i for i, p in enumerate(paths_array) if p in train_paths]
            val_idx = [i for i, p in enumerate(paths_array) if p in val_paths]
            test_idx = [i for i, p in enumerate(paths_array) if p in test_paths]
            
            result['train']['metadata'] = metadata_array[train_idx].tolist()
            result['val']['metadata'] = metadata_array[val_idx].tolist()
            result['test']['metadata'] = metadata_array[test_idx].tolist()
        
        logger.info(f"Dataset split: train={len(train_paths)}, "
                   f"val={len(val_paths)}, test={len(test_paths)}")
        
        return result
    
    def save_split_info(
        self,
        split_data: Dict,
        output_path: Union[str, Path]
    ) -> None:
        """Save dataset split information to JSON file.
        
        Args:
            split_data: Output from create_dataset_split
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        logger.info(f"Split info saved to {output_path}")
    
    def load_split_info(
        self,
        input_path: Union[str, Path]
    ) -> Dict:
        """Load dataset split information from JSON file.
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            Dictionary with train/val/test split information
        """
        with open(input_path, 'r') as f:
            split_data = json.load(f)
        
        logger.info(f"Split info loaded from {input_path}")
        return split_data
    
    def compute_dataset_statistics(
        self,
        dicom_paths: List[Union[str, Path]],
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Compute mean and std across dataset for standardization.
        
        Args:
            dicom_paths: List of DICOM file paths
            sample_size: Number of images to sample (None = all)
            
        Returns:
            Dictionary with 'mean' and 'std' keys
        """
        if sample_size and sample_size < len(dicom_paths):
            indices = np.random.choice(len(dicom_paths), sample_size, replace=False)
            sample_paths = [dicom_paths[i] for i in indices]
        else:
            sample_paths = dicom_paths
        
        pixel_values = []
        for path in sample_paths:
            pixel_array = self.load_dicom_image(path)
            windowed = self.apply_windowing(pixel_array)
            pixel_values.append(windowed.flatten())
        
        all_pixels = np.concatenate(pixel_values)
        
        stats = {
            'mean': float(all_pixels.mean()),
            'std': float(all_pixels.std()),
            'min': float(all_pixels.min()),
            'max': float(all_pixels.max())
        }
        
        self.mean = stats['mean']
        self.std = stats['std']
        
        logger.info(f"Dataset statistics: mean={stats['mean']:.2f}, "
                   f"std={stats['std']:.2f}")
        
        return stats
