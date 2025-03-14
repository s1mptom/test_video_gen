"""LUT Builder for YUV to RGB mapping."""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2

from src.pattern_generator import PatternGenerator
from src.visual_validation.visual_validator import VisualValidationProcessor
from src.pattern_metadata import PatternMetadataHandler
from src.utils.constants import MARKER_PATCHES


class LutBuilder:
    """Class for building 3D LUT from calibration frames."""
    
    def __init__(
        self, 
        metadata_path: str, 
        output_dir: str = "output"
    ):
        """
        Initialize LUT Builder.
        
        Args:
            metadata_path: Path to the pattern metadata JSON file
            output_dir: Directory for output files
        """
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LUT data storage (3D array flattened to 1D for efficient access)
        # Format: Y (8-bit) + U (8-bit) + V (8-bit) -> R, G, B (each 8-bit)
        self.lut_data = np.zeros((256 * 256 * 256 * 3), dtype=np.uint8)
        
        # Tracking variables
        self.processed_frames = 0
        self.processed_patterns = set()
        self.lut_complete = False
        self.pattern_generator_initialized = False
        
        # Debug directory
        self.debug_dir = self.output_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)
        
        # Pattern generator will be initialized after loading metadata
        self.pattern_generator = None
        
        # Metadata handler
        self.metadata_handler = PatternMetadataHandler(output_dir=output_dir)
        
        # Visual validator for marker detection
        self.visual_validator = VisualValidationProcessor(
            output_dir=output_dir, 
            debug_mode=True, 
            debug_dir=self.debug_dir
        )
        
        # Load pattern metadata
        print(f"Loading metadata from {metadata_path}")
        if os.path.exists(metadata_path):
            self.metadata = self.metadata_handler.load_all_metadata()
            print(f"Loaded metadata for {len(self.metadata)} patterns")
            
            # Extract configuration from metadata
            self._extract_config_from_metadata()
        else:
            print(f"Warning: Metadata file not found: {metadata_path}")
            self.metadata = []
            # Set default configuration
            self.config = {
                "width": 1920,
                "height": 1080,
                "patch_size": 16,
                "patch_gap": 4,
                "color_range_percent": 10.0,
                "bit_depth": 8
            }
        
        # Create a progress tracking file
        self.progress_file = self.output_dir / "lut_progress.json"
        self._load_progress()
        
    def _extract_config_from_metadata(self):
        """Extract configuration parameters from metadata."""
        # Try to find config in a dedicated metadata entry or in the first pattern
        config = None
        
        # First check if there's a dedicated config entry
        for entry in self.metadata:
            if entry.get("type") == "config" or entry.get("content_type") == "config":
                config = entry
                break
        
        # If not found, try to get config from the first pattern
        if not config and self.metadata:
            first_pattern = self.metadata[0]
            if "config" in first_pattern:
                config = first_pattern["config"]
            elif "generator_config" in first_pattern:
                config = first_pattern["generator_config"]
        
        # If still not found, create a new one and try to extract parameters from patterns
        if not config:
            config = {}
            
            # Try to infer some parameters from pattern data
            if self.metadata:
                first_pattern = self.metadata[0]
                
                # Try to extract dimensions from patch coordinates
                patches = first_pattern.get("patches", [])
                if patches:
                    max_x = max_y = 0
                    for patch in patches:
                        if "x_range" in patch and len(patch["x_range"]) == 2:
                            max_x = max(max_x, patch["x_range"][1])
                        if "y_range" in patch and len(patch["y_range"]) == 2:
                            max_y = max(max_y, patch["y_range"][1])
                    
                    if max_x > 0 and max_y > 0:
                        config["width"] = max_x + 50  # Add margin
                        config["height"] = max_y + 50
                
                # Try to infer color range from colors
                colors = first_pattern.get("colors", [])
                if colors:
                    unique_values = set()
                    for color in colors:
                        for component in color:
                            unique_values.add(component)
                    
                    unique_count = len(unique_values)
                    if unique_count > 0:
                        levels = max(int(unique_count ** (1/3)), 1)
                        config["color_range_percent"] = (levels / 256) * 100
                        print(f"Estimated color range: {config['color_range_percent']:.1f}% (detected {levels} levels)")
        
        # Fill missing values with defaults
        self.config = {
            "width": config.get("width", 1920),
            "height": config.get("height", 1080),
            "patch_size": config.get("patch_size", 16),
            "patch_gap": config.get("patch_gap", 4),
            "color_range_percent": config.get("color_range_percent", 10.0),
            "bit_depth": config.get("bit_depth", 8)
        }
        
        print(f"Using configuration: {self.config}")
        
        # Initialize pattern generator with extracted config
        self.pattern_generator = PatternGenerator(
            width=self.config["width"],
            height=self.config["height"],
            patch_size=self.config["patch_size"],
            patch_gap=self.config["patch_gap"],
            color_range_percent=self.config["color_range_percent"],
            bit_depth=self.config["bit_depth"]
        )
        
        self.pattern_generator_initialized = True
    
    def _load_progress(self):
        """Load progress from previous run if available."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.processed_patterns = set(progress.get("processed_patterns", []))
                    self.processed_frames = progress.get("processed_frames", 0)
                    print(f"Loaded progress: {len(self.processed_patterns)} patterns processed")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading progress: {e}")
    
    """
    Исправления для метода process_frame в LutBuilder,
    чтобы не учитывать запись конфигурации как паттерн
    """

    def process_frame(self, y_data: np.ndarray, uv_data: np.ndarray, width: int, height: int) -> bool:
        """
        Process a received NV12 frame.
        
        Args:
            y_data: Y plane data
            uv_data: UV plane data
            width: Frame width
            height: Frame height
            
        Returns:
            bool: True if the frame was successfully processed, False otherwise
        """
        # Check if LUT is already complete
        if self.lut_complete:
            print("LUT is already complete. No more frames needed.")
            return False
        
        # Convert NV12 to planar YUV
        yuv_frame = self._nv12_to_planar_yuv(y_data, uv_data, width, height)
        
        # Save frame for debugging
        self._save_debug_frame(yuv_frame, f"received_frame_{self.processed_frames}")
        
        # Проверка инициализации pattern_generator
        if not self.pattern_generator:
            print("Pattern generator not initialized. Attempting to initialize...")
            self._extract_config_from_metadata()
            
            # Проверка после инициализации
            if not self.pattern_generator:
                print("CRITICAL ERROR: Failed to initialize pattern generator!")
                print("Creating pattern generator with default parameters as fallback")
                self.config = {
                    "width": width,
                    "height": height,
                    "patch_size": 16,
                    "patch_gap": 4,
                    "color_range_percent": 10.0,
                    "bit_depth": 8
                }
                
                # Инициализация с дефолтными параметрами
                self.pattern_generator = PatternGenerator(
                    width=width,
                    height=height,
                    patch_size=16,
                    patch_gap=4,
                    color_range_percent=10.0,
                    bit_depth=8
                )
        
        # Check if we need to update the pattern generator width/height based on the frame
        if width != self.config["width"] or height != self.config["height"]:
            print(f"Frame dimensions ({width}x{height}) don't match config ({self.config['width']}x{self.config['height']})")
            print("Updating pattern generator with new dimensions")
            
            # Update configuration
            self.config["width"] = width
            self.config["height"] = height
            
            # Reinitialize pattern generator
            self.pattern_generator = PatternGenerator(
                width=width,
                height=height,
                patch_size=self.config["patch_size"],
                patch_gap=self.config["patch_gap"],
                color_range_percent=self.config["color_range_percent"],
                bit_depth=self.config["bit_depth"]
            )
        
        # Try to read pattern marker
        try:
            pattern_idx, marker_diagnostics = self.visual_validator.read_pattern_marker(yuv_frame, self.pattern_generator)
        except Exception as e:
            print(f"ERROR reading pattern marker: {e}")
            # Dump pattern generator details for debugging
            print(f"Pattern generator details:")
            print(f"  Width: {self.pattern_generator.width}")
            print(f"  Height: {self.pattern_generator.height}")
            print(f"  Patch size: {self.pattern_generator.patch_size}")
            print(f"  Patch gap: {self.pattern_generator.patch_gap}")
            return False
        
        if pattern_idx == -1:
            print(f"Invalid or unrecognized frame: {marker_diagnostics.get('error', 'Unknown error')}")
            return False
        
        # Check if we've already processed this pattern
        if pattern_idx in self.processed_patterns:
            print(f"Pattern {pattern_idx} already processed, skipping.")
            return True
        
        # Get pattern metadata
        pattern_metadata = self._get_pattern_metadata(pattern_idx)
        if not pattern_metadata:
            print(f"Metadata not found for pattern {pattern_idx}")
            return False
        
        # Extract patch values from the frame
        patches_metadata = pattern_metadata.get("patches", [])
        extracted_values = self.visual_validator.extract_patch_values(yuv_frame, patches_metadata)
        
        # Map YUV values to original RGB colors and update LUT
        colors = pattern_metadata.get("colors", [])
        self._update_lut_from_patches(extracted_values, patches_metadata, colors)
        
        # Mark pattern as processed
        self.processed_patterns.add(pattern_idx)
        self.processed_frames += 1
        print(f"Processed frame with pattern {pattern_idx} (total: {self.processed_frames})")
        
        # Save progress
        self._save_progress()
        
        # Вычисляем количество паттернов, исключая конфигурацию
        actual_patterns = [m for m in self.metadata if 
                        not (m.get("type") == "config" or m.get("content_type") == "config") and
                        "pattern_idx" in m and "patches" in m]
        total_patterns = len(actual_patterns)
        
        # Check if we've processed all patterns (исключая конфигурацию)
        if len(self.processed_patterns) >= total_patterns:
            print(f"All patterns processed ({len(self.processed_patterns)}/{total_patterns}). LUT is complete.")
            self.lut_complete = True
            
            # Save the final LUT
            self.save_lut(str(self.output_dir / "calibration_lut.bin"))
            
            # Save a CSV representation for analysis
            self.save_lut_csv(str(self.output_dir / "calibration_lut.csv"))
        
        return True


    def _save_progress(self):
        """Save current progress."""
        # Вычисляем количество паттернов, исключая конфигурацию
        actual_patterns = [m for m in self.metadata if 
                        not (m.get("type") == "config" or m.get("content_type") == "config") and
                        "pattern_idx" in m and "patches" in m]
        total_patterns = len(actual_patterns)
        
        with open(self.progress_file, 'w') as f:
            json.dump({
                "processed_patterns": list(self.processed_patterns),
                "processed_frames": self.processed_frames,
                "total_patterns": total_patterns
            }, f)
                    
    def _initialize_pattern_generator(self, yuv_frame: Dict[str, np.ndarray], width: int, height: int) -> bool:
        """
        Auto-detect pattern size and gap from the first frame.
        
        Args:
            yuv_frame: The YUV frame
            width: Frame width
            height: Frame height
            
        Returns:
            bool: True if successfully initialized, False otherwise
        """
        try:
            print("Attempting to auto-detect patch size and gap...")
            
            # First, try to find the technical row by scanning for high contrast areas
            y_plane = yuv_frame['Y']
            
            # Look for potential marker by analyzing horizontal stripes in the Y plane
            tech_row = -1
            potential_tech_rows = []
            
            # Analyze horizontal stripes for high contrast (potential technical rows)
            for y in range(0, height, self.default_patch_size // 2):
                if y + self.default_patch_size >= height:
                    continue
                    
                # Get a horizontal stripe
                stripe = y_plane[y:y+self.default_patch_size, :]
                
                # Calculate variance (high variance indicates potential markers)
                variance = np.var(stripe)
                
                # Check for alternating patterns (marker signature)
                # Sample the first potential marker region
                marker_width = self.default_patch_size * MARKER_PATCHES
                if marker_width < width:
                    marker_region = stripe[:, :marker_width]
                    transitions = np.sum(np.abs(np.diff(marker_region.mean(axis=0) > 128)))
                    
                    # More transitions suggests a marker pattern
                    if transitions > 3:
                        potential_tech_rows.append((y, variance, transitions))
            
            # Sort by variance and transitions (higher is better)
            potential_tech_rows.sort(key=lambda x: (x[2], x[1]), reverse=True)
            
            if potential_tech_rows:
                tech_row, variance, transitions = potential_tech_rows[0]
                print(f"Detected potential technical row at y={tech_row} (variance={variance:.1f}, transitions={transitions})")
                
                # Estimate patch size by analyzing the marker pattern
                patch_size = self._estimate_patch_size(yuv_frame, tech_row)
                
                if patch_size > 0:
                    # Estimate patch gap based on first transitions
                    marker_region = y_plane[tech_row:tech_row+patch_size, :width//2]
                    transitions = np.where(np.abs(np.diff(marker_region.mean(axis=0) > 128)))[0]
                    
                    # If we have enough transitions, estimate the gap
                    if len(transitions) >= 3:
                        # Average distance between transitions minus patch size
                        distances = np.diff(transitions)
                        patch_gap = max(1, int(np.median(distances) - patch_size))
                    else:
                        patch_gap = self.default_patch_gap
                    
                    print(f"Auto-detected: patch_size={patch_size}, patch_gap={patch_gap}")
                    
                    # Initialize the pattern generator with detected values
                    self.pattern_generator = PatternGenerator(
                        width=width,
                        height=height,
                        patch_size=patch_size,
                        patch_gap=patch_gap,
                        color_range_percent=self.color_range_percent,
                        bit_depth=8
                    )
                    
                    return True
            
            # If auto-detection failed, print a message
            print("Auto-detection of pattern parameters failed, using defaults")
            return False
        
        except Exception as e:
            print(f"Error during pattern parameter auto-detection: {e}")
            return False
    
    def _estimate_patch_size(self, yuv_frame: Dict[str, np.ndarray], tech_row: int) -> int:
        """
        Estimate patch size by analyzing the marker pattern.
        
        Args:
            yuv_frame: The YUV frame
            tech_row: Detected technical row position
            
        Returns:
            int: Estimated patch size or -1 if detection failed
        """
        try:
            y_plane = yuv_frame['Y']
            height, width = y_plane.shape
            
            # Look for vertical transitions to estimate patch height
            patch_heights = []
            
            # Check multiple columns in the technical row area
            for x in range(0, min(width, 300), 20):  # Sample columns
                if tech_row + 30 >= height:
                    continue
                    
                # Get vertical stripe
                stripe = y_plane[tech_row:tech_row+30, x]
                
                # Find transitions (edges of patches)
                transitions = np.where(np.abs(np.diff(stripe > 128)))[0]
                
                if len(transitions) > 0:
                    # First transition should be the patch height
                    patch_heights.append(transitions[0] + 1)  # +1 because diff reduces size by 1
            
            if patch_heights:
                # Use median to reduce outlier impact
                patch_size = int(np.median(patch_heights))
                
                # Sanity check - patch size should be reasonable
                if 4 <= patch_size <= 64:
                    return patch_size
            
            return -1
        
        except Exception as e:
            print(f"Error estimating patch size: {e}")
            return -1
    
    def _nv12_to_planar_yuv(self, y_data: np.ndarray, uv_data: np.ndarray, width: int, height: int) -> Dict[str, np.ndarray]:
        """
        Convert NV12 semi-planar format to planar YUV.
        
        Args:
            y_data: Y plane data
            uv_data: UV plane data
            width: Frame width
            height: Frame height
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with 'Y', 'U', 'V' planes
        """
        # Check if the data shapes match what we expect for NV12
        if len(y_data) != width * height:
            print(f"Warning: Y data size {len(y_data)} doesn't match expected size {width * height}")
            
        expected_uv_size = width * height // 2
        if len(uv_data) != expected_uv_size:
            print(f"Warning: UV data size {len(uv_data)} doesn't match expected size {expected_uv_size}")
        
        # Reshape Y data
        y_plane = y_data.reshape((height, width))
        
        # Process UV data (NV12 has interleaved U and V)
        uv_height = height // 2
        uv_width = width // 2
        
        # Deinterleave UV data (NV12 format has UV pairs)
        uv_data = uv_data.reshape((uv_height, width))
        u_plane = np.zeros((uv_height, uv_width), dtype=np.uint8)
        v_plane = np.zeros((uv_height, uv_width), dtype=np.uint8)
        
        # Extract U (even columns) and V (odd columns)
        u_plane = uv_data[:, 0::2]
        v_plane = uv_data[:, 1::2]
        
        return {'Y': y_plane, 'U': u_plane, 'V': v_plane}
    
    def _get_pattern_metadata(self, pattern_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific pattern.
        
        Args:
            pattern_idx: Pattern index
            
        Returns:
            Optional[Dict[str, Any]]: Pattern metadata or None if not found
        """
        for pattern in self.metadata:
            if pattern.get("pattern_idx") == pattern_idx:
                return pattern
        return None
    
    def _update_lut_from_patches(
        self, 
        extracted_values: List[Tuple[float, float, float]], 
        patches_metadata: List[Dict[str, Any]],
        colors: List[List[int]]
    ) -> None:
        """
        Update LUT based on extracted YUV values and original RGB colors.
        
        Args:
            extracted_values: List of extracted (Y, U, V) values
            patches_metadata: List of patch metadata
            colors: List of original RGB colors for the pattern
        """
        updates_count = 0
        skipped_count = 0
        
        for i, (y_val, u_val, v_val) in enumerate(extracted_values):
            if i >= len(patches_metadata):
                continue
            
            # Проверяем на NaN или недопустимые значения
            if np.isnan(y_val) or np.isnan(u_val) or np.isnan(v_val):
                skipped_count += 1
                if skipped_count < 5 or skipped_count % 50 == 0:
                    print(f"Skipping NaN YUV values for patch {i}: ({y_val}, {u_val}, {v_val})")
                continue
                
            patch = patches_metadata[i]
            
            # Get original RGB color from colors list based on index
            # If patch has a color_idx field, use that to index into colors
            # Otherwise, use the sequential index
            if "color_idx" in patch and patch["color_idx"] < len(colors):
                color_idx = patch["color_idx"]
                r, g, b = colors[color_idx]
            elif i < len(colors):
                r, g, b = colors[i]
            else:
                continue
            
            try:
                # Convert float YUV to int indices for LUT
                y_idx = int(round(y_val))
                u_idx = int(round(u_val))
                v_idx = int(round(v_val))
                
                # Ensure indices are within valid range
                y_idx = max(0, min(y_idx, 255))
                u_idx = max(0, min(u_idx, 255))
                v_idx = max(0, min(v_idx, 255))
                
                # For debug output
                if updates_count < 5 or updates_count % 50 == 0:
                    print(f"Mapping YUV({y_idx}, {u_idx}, {v_idx}) → RGB({r}, {g}, {b})")
                
                # Update LUT entry
                self.update_lut_entry(y_idx, u_idx, v_idx, r, g, b)
                updates_count += 1
            except (ValueError, TypeError, OverflowError) as e:
                skipped_count += 1
                if skipped_count < 5:
                    print(f"Error processing YUV values ({y_val}, {u_val}, {v_val}): {e}")
        
        print(f"Updated {updates_count} entries in the LUT, skipped {skipped_count} invalid values")
    
    def update_lut_entry(self, y: int, u: int, v: int, r: int, g: int, b: int) -> None:
        """
        Update a specific entry in the LUT.
        
        Args:
            y, u, v: YUV values (indices in the LUT)
            r, g, b: RGB values to store
        """
        try:
            # Calculate the index in the flattened LUT data
            # Format: y + (u << 8) + (v << 16)
            index = (y + (u << 8) + (v << 16)) * 3
            
            # Проверка выхода за границы массива
            if index < 0 or index >= len(self.lut_data) - 2:
                print(f"Warning: LUT index out of bounds: {index} (max: {len(self.lut_data) - 3})")
                return
                
            # Update the RGB values
            self.lut_data[index] = r
            self.lut_data[index + 1] = g
            self.lut_data[index + 2] = b
        except Exception as e:
            print(f"Error updating LUT entry for YUV({y}, {u}, {v}): {e}")
    
    def get_lut_entry(self, y: int, u: int, v: int) -> Tuple[int, int, int]:
        """
        Get RGB values for specific YUV indices.
        
        Args:
            y, u, v: YUV values
            
        Returns:
            Tuple[int, int, int]: RGB values
        """
        index = (y + (u << 8) + (v << 16)) * 3
        return (
            self.lut_data[index],
            self.lut_data[index + 1],
            self.lut_data[index + 2]
        )
    
    def save_lut(self, output_path: str) -> None:
        """
        Save the LUT to a binary file.
        
        Args:
            output_path: Path to save the LUT
        """
        # Save the raw LUT data
        with open(output_path, 'wb') as f:
            f.write(self.lut_data.tobytes())
        
        print(f"LUT saved to {output_path}")
        print(f"LUT size: {len(self.lut_data)} bytes")
    
    def save_lut_csv(self, output_path: str, sample_count: int = 1000) -> None:
        """
        Save a sample of LUT entries to CSV for analysis.
        
        Args:
            output_path: Path to save the CSV
            sample_count: Number of samples to include
        """
        with open(output_path, 'w') as f:
            f.write("Y,U,V,R,G,B\n")
            
            # Sample random entries
            total_entries = 256 * 256 * 256
            sample_indices = np.random.choice(total_entries, min(sample_count, total_entries), replace=False)
            
            for idx in sample_indices:
                y = idx & 0xFF
                u = (idx >> 8) & 0xFF
                v = (idx >> 16) & 0xFF
                
                r, g, b = self.get_lut_entry(y, u, v)
                f.write(f"{y},{u},{v},{r},{g},{b}\n")
        
        print(f"LUT sample saved to {output_path}")
    
    def _save_debug_frame(self, frame: Dict[str, np.ndarray], name: str) -> None:
        """
        Save a frame for debugging purposes.
        
        Args:
            frame: Dictionary with Y, U, V planes
            name: Base name for the saved file
        """
        try:
            h, w = frame['Y'].shape
            u_resized = cv2.resize(frame['U'], (w, h), interpolation=cv2.INTER_NEAREST)
            v_resized = cv2.resize(frame['V'], (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Convert YUV to RGB for visualization
            yuv = np.stack([frame['Y'], u_resized, v_resized], axis=-1).astype(np.float32)
            
            # Normalize YUV values to 0-1 range
            yuv[:,:,0] = (yuv[:,:,0] - 16) / 219
            yuv[:,:,1] = (yuv[:,:,1] - 128) / 112
            yuv[:,:,2] = (yuv[:,:,2] - 128) / 112
            
            # YUV to RGB conversion matrix (BT.709)
            m = np.array([
                [1.0, 0.0, 1.5748],
                [1.0, -0.1873, -0.4681],
                [1.0, 1.8556, 0.0]
            ])
            
            rgb = np.zeros(yuv.shape, dtype=np.float32)
            rgb[:,:,0] = np.clip(yuv[:,:,0] + m[0,2] * yuv[:,:,2], 0, 1) * 255
            rgb[:,:,1] = np.clip(yuv[:,:,0] + m[1,1] * yuv[:,:,1] + m[1,2] * yuv[:,:,2], 0, 1) * 255
            rgb[:,:,2] = np.clip(yuv[:,:,0] + m[2,1] * yuv[:,:,1], 0, 1) * 255
            
            # Save RGB image
            cv2.imwrite(str(self.debug_dir / f"{name}.png"), rgb.astype(np.uint8))
        except Exception as e:
            print(f"Error saving debug frame: {e}")