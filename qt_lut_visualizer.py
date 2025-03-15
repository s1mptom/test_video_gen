#!/usr/bin/env python3
"""
LUT Visualizer with RGB LUT as default and Numba acceleration for performance.
"""

import socket
import flatbuffers
import numpy as np
import argparse
import os
import sys
import threading
import queue
from pathlib import Path
import cv2
from datetime import datetime
import time

# Add Numba for acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("[*] Numba acceleration available")
except ImportError:
    NUMBA_AVAILABLE = False
    print("[*] Numba not found - install with 'pip install numba' for better performance")

# Qt imports
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal, QObject

# FlatBuffers imports
from hyperionnet.Request import Request
from hyperionnet.Image import Image
from hyperionnet.RawImage import RawImage
from hyperionnet.NV12Image import NV12Image
from hyperionnet.Register import Register
from hyperionnet.Reply import Reply, ReplyStart, ReplyEnd, ReplyAddVideo, ReplyAddRegistered

# Import for YUV to RGB conversion
from src.utils.color_transforms import yuv_to_rgb_bt709


# Signal for thread communication with Qt
class FrameSignals(QObject):
    new_frame = Signal(np.ndarray, np.ndarray)  # Original, LUT-applied


# Global variables
FRAME_QUEUE = queue.Queue(maxsize=10)
QT_APP = QApplication.instance() or QApplication(sys.argv)
SIGNALS = FrameSignals()
MAIN_WINDOW = None
LUT_DATA = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fast LUT Visualizer with Numba Acceleration")
    parser.add_argument("--lut-path", type=str, default="output/calibration_lut.bin",
                      help="Path to the LUT binary file")
    parser.add_argument("--port", type=int, default=9999,
                      help="Port to listen for incoming frames")
    parser.add_argument("--save-frames", action="store_true",
                      help="Save received frames to disk")
    return parser.parse_args()


# Numba-accelerated LUT application function
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def apply_lut_numba(y_plane, u_resized, v_resized, lut_data, height, width):
        """
        Apply LUT using Numba acceleration.
        
        Args:
            y_plane: Y plane data
            u_resized: Resized U plane data
            v_resized: Resized V plane data
            lut_data: The LUT data array
            height, width: Image dimensions
            
        Returns:
            np.ndarray: RGB image with LUT applied
        """
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Process in parallel for better performance
        for y in prange(height):
            for x in range(width):
                # Get YUV values
                y_val = y_plane[y, x]
                u_val = u_resized[y, x]
                v_val = v_resized[y, x]
                
                # Calculate LUT index - LUT data is in RGB order but we need BGR for OpenCV
                idx = (y_val + (u_val << 8) + (v_val << 16)) * 3
                
                # Check bounds
                if idx < len(lut_data) - 2:
                    # LUT data is in RGB order, but we need BGR for OpenCV
                    result[y, x, 0] = lut_data[idx+2]  # B from R
                    result[y, x, 1] = lut_data[idx+1]  # G from G
                    result[y, x, 2] = lut_data[idx]    # R from B
        
        return result


def start_server(args):
    """Start the server."""
    global MAIN_WINDOW, LUT_DATA
    
    # Load LUT
    print("Loading LUT data...")
    lut_path = Path(args.lut_path)
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f"LUT file not found: {lut_path}")
    
    # Load the binary LUT data
    with open(lut_path, 'rb') as f:
        LUT_DATA = np.frombuffer(f.read(), dtype=np.uint8)
    
    print(f"LUT loaded: {len(LUT_DATA)} bytes")
    print("[*] Using RGB LUT format (default)")
    
    # Create frames directory if saving frames
    frames_dir = None
    if args.save_frames:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize Qt window
    MAIN_WINDOW = LutWindow(SIGNALS)
    MAIN_WINDOW.show()
    
    # Connect signals
    SIGNALS.new_frame.connect(MAIN_WINDOW.update_frames)
    
    # Start frame processing thread
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Server setup
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = args.port
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")
    
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=run_server_loop,
        args=(server_socket, frames_dir)
    )
    server_thread.daemon = True
    server_thread.start()
    
    # Run Qt event loop in the main thread
    try:
        QT_APP.exec()
    except KeyboardInterrupt:
        print("\n[*] Stopped by user")
    finally:
        os._exit(0)  # Force exit all threads


def run_server_loop(server_socket, frames_dir):
    """Run the server loop in a separate thread."""
    try:
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"[+] Accepted connection from {client_address}")
            
            # Handle client in a new thread
            client_thread = threading.Thread(
                target=process_client,
                args=(client_socket, frames_dir)
            )
            client_thread.daemon = True
            client_thread.start()
            
    except Exception as e:
        print(f"[-] Server error: {e}")
    finally:
        server_socket.close()


def process_client(client_socket, frames_dir):
    """Process a client connection."""
    # Stats for tracking
    stats = {"frames_received": 0, "frames_processed": 0}
    
    try:
        while True:
            # Reading message length (4 bytes, big-endian)
            data = client_socket.recv(4)
            if not data:
                break
                
            message_length = int.from_bytes(data, byteorder='big')
            print(f"[*] Message length: {message_length}")
            
            # Reading the message
            data = bytearray()
            while len(data) < message_length:
                packet = client_socket.recv(message_length - len(data))
                if not packet:
                    break
                data.extend(packet)
            
            if len(data) != message_length:
                print(f"[-] Error: Incomplete data received. Expected {message_length}, got {len(data)}")
                break
            
            print(f"[*] Received data of length: {len(data)}")
            
            # Decoding message with FlatBuffers
            try:
                request = Request.GetRootAsRequest(data, 0)
                process_request(request, client_socket, frames_dir, stats)
                    
            except Exception as e:
                print(f"[-] Error processing request: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"[-] Error handling client: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client_socket.close()
        print(f"[*] Client connection closed")


def process_request(request, client_socket, frames_dir, stats):
    """Process a FlatBuffers request."""
    command_type = request.CommandType()
    print(f"[*] Command type: {command_type}")
    
    if command_type == 2:  # Image command
        process_image(request, client_socket, frames_dir, stats)
    elif command_type == 4:  # Registration
        process_registration(request, client_socket)
    else:
        print(f"[*] Unknown command type: {command_type}")


def process_image(request, client_socket, frames_dir, stats):
    """Process an image command."""
    command = request.Command()
    image = Image()
    image.Init(command.Bytes, command.Pos)
    
    # Increment frames received counter
    stats["frames_received"] += 1
    
    image_type = image.DataType()
    success = False
    
    if image_type == 1:  # RawImage
        raw_image = RawImage()
        raw_image.Init(image.Data().Bytes, image.Data().Pos)
        width, height = raw_image.Width(), raw_image.Height()
        data = raw_image.DataAsNumpy()
        print(f"[*] Received RawImage with dimensions {width}x{height}")
        
        # Raw image not supported for visualization
        print(f"[-] RawImage format not supported for visualization")
        
        # Optionally save the frame
        if frames_dir:
            frame_path = frames_dir / f"raw_frame_{stats['frames_received']}.raw"
            with open(frame_path, "wb") as f:
                f.write(data.tobytes())
            print(f"[*] Saved raw frame to {frame_path}")
        
    elif image_type == 2:  # NV12Image
        try:
            nv12_image = NV12Image()
            nv12_image.Init(image.Data().Bytes, image.Data().Pos)
            width, height = nv12_image.Width(), nv12_image.Height()
            y_data = nv12_image.YDataAsNumpy()
            uv_data = nv12_image.UvDataAsNumpy()
            print(f"[*] Received NV12Image with dimensions {width}x{height}")
            
            # Add frame to processing queue for display
            FRAME_QUEUE.put((y_data, uv_data, width, height))
            
            # Mark as successfully processed
            success = True
            stats["frames_processed"] += 1
            print(f"[*] Added frame #{stats['frames_processed']} to processing queue")
            
            # Optionally save the frame
            if frames_dir:
                frame_path = frames_dir / f"nv12_frame_{stats['frames_received']}.yuv"
                with open(frame_path, "wb") as f:
                    f.write(y_data.tobytes())
                    f.write(uv_data.tobytes())
                print(f"[*] Saved NV12 frame to {frame_path}")
        except Exception as e:
            print(f"[-] Error processing NV12 image: {e}")
            import traceback
            traceback.print_exc()
    
    # Send response
    try:
        builder = flatbuffers.Builder(0)
        
        # Create Reply object
        ReplyStart(builder)
        video_status = 1 if success else 0  # 1 = success, 0 = not processed
        ReplyAddVideo(builder, video_status)
        ReplyAddRegistered(builder, -1)  # Not used for images
        reply = ReplyEnd(builder)
        
        builder.Finish(reply)
        
        data = builder.Output()
        message_size = len(data)
        
        # Send size as 4-byte big-endian integer
        header = message_size.to_bytes(4, byteorder='big')
        
        client_socket.sendall(header + data)
        print(f"[*] Sent image acknowledgment of length: {message_size + 4}")
    except Exception as e:
        print(f"[-] Error sending image acknowledgment: {e}")
        import traceback
        traceback.print_exc()


def process_registration(request, client_socket):
    """Process a registration command."""
    command = request.Command()
    register_command = Register()
    register_command.Init(command.Bytes, command.Pos)
    
    origin = register_command.Origin().decode('utf-8')
    priority = register_command.Priority()
    
    print(f"[*] Registration request: Origin = {origin}, Priority = {priority}")
    
    # Send registration acknowledgment
    send_acknowledgment(client_socket, priority)


def send_acknowledgment(client_socket, priority):
    """Send a registration acknowledgment."""
    builder = flatbuffers.Builder(0)
    
    # Create a Reply object
    ReplyStart(builder)
    ReplyAddVideo(builder, -1)
    ReplyAddRegistered(builder, priority)
    reply = ReplyEnd(builder)
    
    builder.Finish(reply)
    
    data = builder.Output()
    message_size = len(data)
    
    # Send the size as a 4-byte big-endian integer
    header = message_size.to_bytes(4, byteorder='big')
    
    client_socket.sendall(header + data)
    print(f"[*] Sent acknowledgment of length: {message_size + 4}")


def process_frames():
    """Process frames from the queue and display them."""
    try:
        while True:
            try:
                # Get frame from queue with timeout
                y_data, uv_data, width, height = FRAME_QUEUE.get(timeout=1.0)
                
                # Convert to RGB for display - using fast vectorized approach
                start_time = time.time()
                standard_rgb, lut_rgb = convert_frame_fast(y_data, uv_data, width, height)
                processing_time = time.time() - start_time
                
                print(f"[*] Frame conversion completed in {processing_time:.3f} seconds")
                
                # Send to UI
                SIGNALS.new_frame.emit(standard_rgb, lut_rgb)
                
            except queue.Empty:
                # No frame in queue, just wait
                pass
                
    except Exception as e:
        print(f"[-] Frame processing error: {e}")
        import traceback
        traceback.print_exc()


def apply_lut_fast(y_plane, u_resized, v_resized):
    """
    Apply LUT to YUV values using the fastest approach available.
    
    Args:
        y_plane: Y plane data
        u_resized: Resized U plane data
        v_resized: Resized V plane data
        
    Returns:
        np.ndarray: RGB image with LUT applied
    """
    height, width = y_plane.shape
    
    # Use Numba if available
    if NUMBA_AVAILABLE:
        return apply_lut_numba(y_plane, u_resized, v_resized, LUT_DATA, height, width)
    
    # Fallback to vectorized NumPy if Numba is not available
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100000
    total_pixels = height * width
    
    # Reshape the data for faster access
    y_flat = y_plane.flatten().astype(np.uint32)
    u_flat = u_resized.flatten().astype(np.uint32)
    v_flat = v_resized.flatten().astype(np.uint32)
    
    for start_idx in range(0, total_pixels, batch_size):
        end_idx = min(start_idx + batch_size, total_pixels)
        
        # Extract batch
        y_batch = y_flat[start_idx:end_idx]
        u_batch = u_flat[start_idx:end_idx]
        v_batch = v_flat[start_idx:end_idx]
        
        # Calculate indices
        indices = (y_batch + (u_batch << 8) + (v_batch << 16)) * 3
        
        # Apply LUT to valid indices
        valid_mask = (indices < len(LUT_DATA) - 2)
        
        for i in range(len(indices)):
            if valid_mask[i]:
                idx = indices[i]
                
                # Calculate pixel coordinates
                pixel_idx = start_idx + i
                y_idx = pixel_idx // width
                x_idx = pixel_idx % width
                
                # LUT data is in RGB order, but we need BGR for OpenCV
                result[y_idx, x_idx, 0] = LUT_DATA[idx+2]  # B from R
                result[y_idx, x_idx, 1] = LUT_DATA[idx+1]  # G from G
                result[y_idx, x_idx, 2] = LUT_DATA[idx]    # R from B
    
    return result


def convert_frame_fast(y_data, uv_data, width, height):
    """
    Convert YUV frame to RGB using the fastest approach available.
    
    Args:
        y_data: Y plane data
        uv_data: UV plane data
        width: Frame width
        height: Frame height
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Standard RGB image, LUT-applied RGB image
    """
    try:
        # 1. Convert NV12 to planar YUV
        y_plane = y_data.reshape((height, width))
        
        # Process UV data (NV12 has interleaved U and V)
        uv_height = height // 2
        
        # Reshape and deinterleave UV data
        uv_data = uv_data.reshape((uv_height, width))
        u_plane = uv_data[:, 0::2]
        v_plane = uv_data[:, 1::2]
        
        # 2. Resize UV planes to Y plane size
        u_resized = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_NEAREST)
        v_resized = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # 3. Convert to RGB using standard formula
        # yuv_to_rgb_bt709 returns BGR order for OpenCV
        standard_rgb = yuv_to_rgb_bt709(y_plane, u_resized, v_resized)
        
        # 4. Apply LUT using the fastest available method
        start_lut = time.time()
        lut_rgb = apply_lut_fast(y_plane, u_resized, v_resized)
        lut_time = time.time() - start_lut
        print(f"[*] LUT application took {lut_time:.3f} seconds")
        
        # Add labels to images - OpenCV text rendering works with BGR
        cv2.putText(standard_rgb, "Original", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(lut_rgb, "LUT Applied", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return standard_rgb, lut_rgb
        
    except Exception as e:
        print(f"[-] Error converting frame: {e}")
        import traceback
        traceback.print_exc()
        # Return blank images on error
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return blank, blank


class LutWindow(QMainWindow):
    """Qt window for displaying original and LUT-applied frames."""
    
    def __init__(self, signals):
        super().__init__()
        
        # Set up UI
        self.setWindowTitle("LUT Visualizer")
        self.resize(1280, 720)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create frame display layout
        frame_layout = QHBoxLayout()
        main_layout.addLayout(frame_layout)
        
        # Create labels for displaying frames
        self.original_label = QLabel("Waiting for frames...")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(640, 480)
        
        self.lut_label = QLabel("Waiting for frames...")
        self.lut_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lut_label.setMinimumSize(640, 480)
        
        frame_layout.addWidget(self.original_label)
        frame_layout.addWidget(self.lut_label)
        
        print("[*] LUT Window initialized")
    
    def update_frames(self, original_frame, lut_frame):
        """Update frame display with received frames."""
        try:
            # Important note: OpenCV uses BGR order, but Qt needs RGB
            # Convert from BGR to RGB for Qt display
            original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            lut_rgb = cv2.cvtColor(lut_frame, cv2.COLOR_BGR2RGB)
            
            # Get dimensions
            h, w, c = original_rgb.shape
            
            # Create QImage
            bytes_per_line = w * c
            original_qimg = QImage(original_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            lut_qimg = QImage(lut_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create pixmaps
            original_pixmap = QPixmap.fromImage(original_qimg)
            lut_pixmap = QPixmap.fromImage(lut_qimg)
            
            # Scale to fit labels
            original_pixmap = original_pixmap.scaled(
                self.original_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            lut_pixmap = lut_pixmap.scaled(
                self.lut_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Update labels
            self.original_label.setPixmap(original_pixmap)
            self.lut_label.setPixmap(lut_pixmap)
            
            print("[*] Updated frame display")
        except Exception as e:
            print(f"[-] Error updating frames: {e}")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Handle window close event."""
        print("[*] Window close event received")
        event.accept()
        os._exit(0)  # Force exit all threads


if __name__ == "__main__":
    args = parse_args()
    start_server(args)