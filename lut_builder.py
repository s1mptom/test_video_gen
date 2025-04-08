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
from src.utils.constants import (
    MARKER_PATCHES, ChromaFormat, ColorRange, 
    get_yuv_constants
)


class LutBuilder:
    """Class for building 3D LUT from calibration frames."""
    
    def __init__(
        self, 
        metadata_path: str, 
        output_dir: str = "output",
        chroma_format: str = None,
        color_range: str = None
    ):
        """
        Initialize LUT Builder.
        
        Args:
            metadata_path: Path to the pattern metadata JSON file
            output_dir: Directory for output files
            chroma_format: Chroma subsampling format (optional, can be detected from metadata)
            color_range: Color range (optional, can be detected from metadata)
        """
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set initial format and range (will be updated from metadata if not provided)
        self.chroma_format = chroma_format
        self.color_range = color_range
        
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
                "bit_depth": 8,
                "chroma_format": self.chroma_format or ChromaFormat.YUV_422,
                "color_range": self.color_range or ColorRange.LIMITED
            }
            
            # Update format and range if not provided
            if self.chroma_format is None:
                self.chroma_format = self.config["chroma_format"]
            if self.color_range is None:
                self.color_range = self.config["color_range"]
        
        print(f"Using chroma format: {self.chroma_format}, color range: {self.color_range}")
        
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
                config = entry.get("config", entry)
                break
        
        # If not found, try to get config from the first pattern
        if not config and self.metadata:
            first_pattern = self.metadata[0]
            if "config" in first_pattern:
                config = first_pattern["config"]
            elif "generator_config" in first_pattern:
                config = first_pattern["generator_config"]
            # Check for format info directly in the pattern
            elif "format" in first_pattern:
                pattern_format = first_pattern["format"]
                if self.chroma_format is None and "chroma_subsampling" in pattern_format:
                    self.chroma_format = pattern_format["chroma_subsampling"]
                if self.color_range is None and "color_range" in pattern_format:
                    self.color_range = pattern_format["color_range"]
        
        # If still not found, create a new one and try to extract parameters from patterns
        if not config:
            config = {}
            
            # Try to infer some parameters from pattern data
            if self.metadata:
                first_pattern = self.metadata[0]
                
                # Check for format info
                if "format" in first_pattern:
                    format_info = first_pattern["format"]
                    if self.chroma_format is None and "chroma_subsampling" in format_info:
                        self.chroma_format = format_info["chroma_subsampling"]
                    if self.color_range is None and "color_range" in format_info:
                        self.color_range = format_info["color_range"]
                
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
            "bit_depth": config.get("bit_depth", 8),
            "chroma_format": config.get("chroma_format", self.chroma_format or ChromaFormat.YUV_422),
            "color_range": config.get("color_range", self.color_range or ColorRange.LIMITED)
        }
        
        # Update format and range if not provided
        if self.chroma_format is None:
            self.chroma_format = self.config["chroma_format"]
        if self.color_range is None:
            self.color_range = self.config["color_range"]
        
        print(f"Using configuration: {self.config}")
        
        # Initialize pattern generator with extracted config
        self.pattern_generator = PatternGenerator(
            width=self.config["width"],
            height=self.config["height"],
            patch_size=self.config["patch_size"],
            patch_gap=self.config["patch_gap"],
            color_range_percent=self.config["color_range_percent"],
            bit_depth=self.config["bit_depth"],
            chroma_subsampling=self.chroma_format,
            color_range=self.color_range
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
                    
                    # Load format and range if available
                    if "chroma_format" in progress and self.chroma_format is None:
                        self.chroma_format = progress["chroma_format"]
                    if "color_range" in progress and self.color_range is None:
                        self.color_range = progress["color_range"]
                        
                    print(f"Loaded progress: {len(self.processed_patterns)} patterns processed")
                    print(f"Loaded format: {self.chroma_format}, range: {self.color_range}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading progress: {e}")

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
        
        # Convert NV12 to planar YUV based on format
        yuv_frame = self._nv12_to_planar_yuv(y_data, uv_data, width, height, self.chroma_format)
        
        # Save frame for debugging
        self._save_debug_frame(yuv_frame, f"received_frame_{self.processed_frames}")
        
        # Check if pattern generator is initialized
        if not self.pattern_generator:
            print("Pattern generator not initialized. Attempting to initialize...")
            self._extract_config_from_metadata()
            
            # Check after initialization
            if not self.pattern_generator:
                print("CRITICAL ERROR: Failed to initialize pattern generator!")
                print("Creating pattern generator with default parameters as fallback")
                self.config = {
                    "width": width,
                    "height": height,
                    "patch_size": 16,
                    "patch_gap": 4,
                    "color_range_percent": 10.0,
                    "bit_depth": 8,
                    "chroma_format": self.chroma_format or ChromaFormat.YUV_422,
                    "color_range": self.color_range or ColorRange.LIMITED
                }
                
                # Initialize with default parameters
                self.pattern_generator = PatternGenerator(
                    width=width,
                    height=height,
                    patch_size=16,
                    patch_gap=4,
                    color_range_percent=10.0,
                    bit_depth=8,
                    chroma_subsampling=self.chroma_format,
                    color_range=self.color_range
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
                bit_depth=self.config["bit_depth"],
                chroma_subsampling=self.chroma_format,
                color_range=self.color_range
            )
        
        # Try to read pattern marker
        try:
            pattern_idx, marker_diagnostics = self.visual_validator.read_pattern_marker(
                yuv_frame, self.pattern_generator, self.color_range)
        except Exception as e:
            print(f"ERROR reading pattern marker: {e}")
            # Dump pattern generator details for debugging
            print(f"Pattern generator details:")
            print(f"  Width: {self.pattern_generator.width}")
            print(f"  Height: {self.pattern_generator.height}")
            print(f"  Patch size: {self.pattern_generator.patch_size}")
            print(f"  Patch gap: {self.pattern_generator.patch_gap}")
            print(f"  Chroma format: {self.chroma_format}")
            print(f"  Color range: {self.color_range}")
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
        
        # Check format and range in metadata
        if "format" in pattern_metadata:
            pattern_format = pattern_metadata["format"]
            pattern_chroma = pattern_format.get("chroma_subsampling")
            pattern_range = pattern_format.get("color_range")
            
            if pattern_chroma and pattern_chroma != self.chroma_format:
                print(f"Warning: Pattern chroma format ({pattern_chroma}) differs from builder format ({self.chroma_format})")
            
            if pattern_range and pattern_range != self.color_range:
                print(f"Warning: Pattern color range ({pattern_range}) differs from builder range ({self.color_range})")
                
            # Use format from pattern if needed
            if pattern_chroma and pattern_range:
                # Extract patch values with the pattern's format and range
                patches_metadata = pattern_metadata.get("patches", [])
                extracted_values = self.visual_validator.extract_patch_values(
                    yuv_frame, patches_metadata, pattern_chroma, pattern_range)
                
                # Map YUV values to original RGB colors and update LUT
                colors = pattern_metadata.get("colors", [])
                self._update_lut_from_patches(extracted_values, patches_metadata, colors, pattern_chroma, pattern_range)
            else:
                # Use default format and range
                patches_metadata = pattern_metadata.get("patches", [])
                extracted_values = self.visual_validator.extract_patch_values(
                    yuv_frame, patches_metadata, self.chroma_format, self.color_range)
                
                # Map YUV values to original RGB colors and update LUT
                colors = pattern_metadata.get("colors", [])
                self._update_lut_from_patches(extracted_values, patches_metadata, colors, 
                                             self.chroma_format, self.color_range)
        else:
            # Use default format and range
            patches_metadata = pattern_metadata.get("patches", [])
            extracted_values = self.visual_validator.extract_patch_values(
                yuv_frame, patches_metadata, self.chroma_format, self.color_range)
            
            # Map YUV values to original RGB colors and update LUT
            colors = pattern_metadata.get("colors", [])
            self._update_lut_from_patches(extracted_values, patches_metadata, colors, 
                                         self.chroma_format, self.color_range)
        
        # Mark pattern as processed
        self.processed_patterns.add(pattern_idx)
        self.processed_frames += 1
        print(f"Processed frame with pattern {pattern_idx} (total: {self.processed_frames})")
        
        # Save progress
        self._save_progress()
        
        # Calculate number of patterns, excluding configuration
        actual_patterns = [m for m in self.metadata if 
                        not (m.get("type") == "config" or m.get("content_type") == "config") and
                        "pattern_idx" in m and "patches" in m]
        total_patterns = len(actual_patterns)
        
        # Check if we've processed all patterns (excluding configuration)
        if len(self.processed_patterns) >= total_patterns:
            print(f"All patterns processed ({len(self.processed_patterns)}/{total_patterns}). LUT is complete.")
            self.lut_complete = True
            
            # Save the final LUT
            self.save_lut(str(self.output_dir / "calibration_lut.bin"))
            
            # Save a CSV representation for analysis
            self.save_lut_csv(str(self.output_dir / "calibration_lut.csv"))
            
            # Also save LUT with format and range info in the filename
            lut_name = f"calibration_lut_{self.chroma_format}_{self.color_range}.bin"
            self.save_lut(str(self.output_dir / lut_name))
        
        return True

    def _save_progress(self):
        """Save current progress."""
        # Calculate number of patterns, excluding configuration
        actual_patterns = [m for m in self.metadata if 
                        not (m.get("type") == "config" or m.get("content_type") == "config") and
                        "pattern_idx" in m and "patches" in m]
        total_patterns = len(actual_patterns)
        
        with open(self.progress_file, 'w') as f:
            json.dump({
                "processed_patterns": list(self.processed_patterns),
                "processed_frames": self.processed_frames,
                "total_patterns": total_patterns,
                "chroma_format": self.chroma_format,
                "color_range": self.color_range
            }, f)
                    
    def _nv12_to_planar_yuv(
        self, 
        y_data: np.ndarray, 
        uv_data: np.ndarray, 
        width: int, 
        height: int, 
        format: str = ChromaFormat.YUV_422
    ) -> Dict[str, np.ndarray]:
        """
        Convert NV12 semi-planar format to planar YUV.
        
        Args:
            y_data: Y plane data
            uv_data: UV plane data
            width: Frame width
            height: Frame height
            format: Target YUV format ("420", "422", or "444")
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with 'Y', 'U', 'V' planes
        """
        # Form Y plane
        y_plane = y_data.reshape((height, width))
        
        # Process UV data (in NV12 U and V are interleaved)
        uv_height = height // 2
        uv_width = width // 2
        
        # First, reshape UV data
        uv_data_reshaped = uv_data.reshape((uv_height, width))
        
        # Deinterleave UV data
        u_plane_420 = uv_data_reshaped[:, 0::2]
        v_plane_420 = uv_data_reshaped[:, 1::2]
        
        # Convert to the requested format
        if format == ChromaFormat.YUV_420:
            # For YUV420, we already have the correct format
            u_plane = u_plane_420
            v_plane = v_plane_420
        elif format == ChromaFormat.YUV_422:
            # For YUV422, we need to upsample vertically
            u_plane = cv2.resize(u_plane_420, (uv_width, height), interpolation=cv2.INTER_LINEAR)
            v_plane = cv2.resize(v_plane_420, (uv_width, height), interpolation=cv2.INTER_LINEAR)
        elif format == ChromaFormat.YUV_444:
            # For YUV444, we need to upsample both horizontally and vertically
            u_plane = cv2.resize(u_plane_420, (width, height), interpolation=cv2.INTER_LINEAR)
            v_plane = cv2.resize(v_plane_420, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Unsupported YUV format: {format}")
        
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
        colors: List[List[int]],
        chroma_format: str = ChromaFormat.YUV_422,
        color_range: str = ColorRange.LIMITED
    ) -> None:
        """
        Update LUT based on extracted YUV values and original RGB colors.
        
        Args:
            extracted_values: List of extracted (Y, U, V) values
            patches_metadata: List of patch metadata
            colors: List of original RGB colors for the pattern
            chroma_format: Chroma subsampling format
            color_range: Color range
        """
        updates_count = 0
        skipped_count = 0
        
        # Create confidence map for this update
        confidence_map = {}
        
        for i, (y_val, u_val, v_val) in enumerate(extracted_values):
            if i >= len(patches_metadata):
                continue
            
            # Check for NaN or invalid values
            if np.isnan(y_val) or np.isnan(u_val) or np.isnan(v_val):
                skipped_count += 1
                continue
                
            patch = patches_metadata[i]
            
            # Get original RGB color
            color_idx = patch.get("color_idx", i)
            if color_idx < len(colors):
                r, g, b = colors[color_idx]
            else:
                continue
            
            try:
                # Convert extracted YUV to LUT indices
                y_idx = int(round(y_val))
                u_idx = int(round(u_val))
                v_idx = int(round(v_val))
                
                # Limit indices to valid range
                y_idx = max(0, min(y_idx, 255))
                u_idx = max(0, min(u_idx, 255))
                v_idx = max(0, min(v_idx, 255))
                
                # Calculate confidence for this sample
                # In 422 and 444 formats we have more color data vertically
                confidence_boost = 1.0
                if chroma_format == ChromaFormat.YUV_422:
                    confidence_boost = 1.2
                elif chroma_format == ChromaFormat.YUV_444:
                    confidence_boost = 1.5
                
                # Assess patch quality
                quality = self._assess_patch_quality(patch, y_val, u_val, v_val)
                confidence = quality * confidence_boost
                
                # Save confidence level
                lut_key = (y_idx, u_idx, v_idx)
                confidence_map[lut_key] = confidence
                
                # Update LUT entry with confidence
                self._update_lut_entry_with_confidence(y_idx, u_idx, v_idx, r, g, b, confidence)
                updates_count += 1
                
            except (ValueError, TypeError, OverflowError) as e:
                skipped_count += 1
                if skipped_count < 5:
                    print(f"Error processing YUV values ({y_val}, {u_val}, {v_val}): {e}")
        
        print(f"Updated {updates_count} LUT entries, skipped {skipped_count} invalid values")
        print(f"Format: {chroma_format}, Range: {color_range}")
        
        # Analyze and smooth LUT for better results
        if updates_count > 0:
            self._analyze_and_smooth_lut(confidence_map)
    
    def _update_lut_entry_with_confidence(
        self, 
        y: int, 
        u: int, 
        v: int, 
        r: int, 
        g: int, 
        b: int, 
        confidence: float = 1.0
    ) -> None:
        """
        Update a LUT entry with confidence weighting.
        
        Args:
            y, u, v: YUV values (indices in the LUT)
            r, g, b: RGB values to store
            confidence: Confidence level (0.0-1.0)
        """
        try:
            # Calculate the index in the flattened LUT data
            index = (y + (u << 8) + (v << 16)) * 3
            
            # Check if index is within bounds
            if index < 0 or index >= len(self.lut_data) - 2:
                return
            
            # If the entry is already populated, use weighted average
            if any(self.lut_data[index:index+3]):
                # Existing values
                r_old = self.lut_data[index]
                g_old = self.lut_data[index + 1]
                b_old = self.lut_data[index + 2]
                
                # Default previous confidence (higher than 0 to avoid overwriting)
                prev_confidence = 0.5
                
                # Weighted average based on confidence
                r_new = int((r * confidence + r_old * prev_confidence) / (confidence + prev_confidence))
                g_new = int((g * confidence + g_old * prev_confidence) / (confidence + prev_confidence))
                b_new = int((b * confidence + b_old * prev_confidence) / (confidence + prev_confidence))
                
                # Update the RGB values
                self.lut_data[index] = r_new
                self.lut_data[index + 1] = g_new
                self.lut_data[index + 2] = b_new
            else:
                # First update for this entry
                self.lut_data[index] = r
                self.lut_data[index + 1] = g
                self.lut_data[index + 2] = b
                
        except Exception as e:
            print(f"Error updating LUT entry for YUV({y}, {u}, {v}): {e}")
            
    def _assess_patch_quality(self, patch, y_val, u_val, v_val):
        """
        Assess patch quality for LUT update weighting.
        
        Args:
            patch: Patch metadata
            y_val, u_val, v_val: Extracted YUV values
            
        Returns:
            float: Quality assessment (0.0-1.0)
        """
        # Quality factors:
        
        # 1. Centrality of YUV values (avoid extreme values)
        y_center = 1.0 - abs(y_val - 128.0) / 128.0
        u_center = 1.0 - abs(u_val - 128.0) / 128.0
        v_center = 1.0 - abs(v_val - 128.0) / 128.0
        centrality = (y_center + u_center + v_center) / 3.0
        
        # 2. Patch size (bigger is better)
        y_range = patch["y_range"]
        x_range = patch["x_range"]
        patch_size = min(y_range[1] - y_range[0], x_range[1] - x_range[0])
        size_quality = min(1.0, patch_size / 16.0)  # Normalize for 16x16 patch
        
        # 3. Frame position (avoid edges)
        frame_height = 1080  # Assumed frame height
        frame_width = 1920   # Assumed frame width
        
        center_y = (y_range[0] + y_range[1]) / 2.0
        center_x = (x_range[0] + x_range[1]) / 2.0
        
        distance_from_center_y = abs(center_y - frame_height/2) / (frame_height/2)
        distance_from_center_x = abs(center_x - frame_width/2) / (frame_width/2)
        position_quality = 1.0 - max(distance_from_center_y, distance_from_center_x)
        
        # Final quality (weighted sum)
        quality = 0.4 * centrality + 0.4 * size_quality + 0.2 * position_quality
        
        # Limit to range [0.1, 1.0]
        return max(0.1, min(1.0, quality))
    
    def _analyze_and_smooth_lut(self, confidence_map):
        """
        Analyze and smooth LUT for better quality.
        
        Args:
            confidence_map: Map of confidence values for updated entries
        """
        # Not fully implemented in this version
        pass
            
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
            
            # Check if index is within bounds
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
        print(f"Format: {self.chroma_format}, Range: {self.color_range}")
    
    def save_lut_csv(self, output_path: str, sample_count: int = 1000) -> None:
        """
        Save a sample of LUT entries to CSV for analysis.
        
        Args:
            output_path: Path to save the CSV
            sample_count: Number of samples to include
        """
        with open(output_path, 'w') as f:
            f.write("Y,U,V,R,G,B,Format,Range\n")
            
            # Sample random entries
            total_entries = 256 * 256 * 256
            sample_indices = np.random.choice(total_entries, min(sample_count, total_entries), replace=False)
            
            for idx in sample_indices:
                y = idx & 0xFF
                u = (idx >> 8) & 0xFF
                v = (idx >> 16) & 0xFF
                
                r, g, b = self.get_lut_entry(y, u, v)
                f.write(f"{y},{u},{v},{r},{g},{b},{self.chroma_format},{self.color_range}\n")
        
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
            
            # Determine format based on UV plane sizes
            format_name = "unknown"
            if frame['U'].shape[0] == h // 2 and frame['U'].shape[1] == w // 2:
                format_name = "420"
            elif frame['U'].shape[0] == h and frame['U'].shape[1] == w // 2:
                format_name = "422"
            elif frame['U'].shape[0] == h and frame['U'].shape[1] == w:
                format_name = "444"
            
            # Resize UV to Y dimensions for visualization
            u_resized = cv2.resize(frame['U'], (w, h), interpolation=cv2.INTER_NEAREST)
            v_resized = cv2.resize(frame['V'], (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Get YUV constants based on color range
            yuv_const = get_yuv_constants(self.color_range)
            y_min = yuv_const["Y_MIN"]
            y_range = yuv_const["Y_RANGE"]
            
            # Convert YUV to RGB for visualization
            yuv = np.stack([frame['Y'], u_resized, v_resized], axis=-1).astype(np.float32)
            
            # Normalize YUV values based on range
            if self.color_range == ColorRange.LIMITED:
                # Limited range (16-235, 16-240)
                yuv[:,:,0] = (yuv[:,:,0] - y_min) / y_range
                yuv[:,:,1] = (yuv[:,:,1] - 128) / 112
                yuv[:,:,2] = (yuv[:,:,2] - 128) / 112
            else:
                # Full range (0-255)
                yuv[:,:,0] = yuv[:,:,0] / 255
                yuv[:,:,1] = (yuv[:,:,1] - 128) / 128
                yuv[:,:,2] = (yuv[:,:,2] - 128) / 128
            
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
            
            # Add format and range info to filename
            range_name = "limited" if self.color_range == ColorRange.LIMITED else "full"
            output_name = f"{name}_{format_name}_{range_name}.png"
            
            # Save RGB image
            cv2.imwrite(str(self.debug_dir / output_name), rgb.astype(np.uint8))
        except Exception as e:
            print(f"Error saving debug frame: {e}")