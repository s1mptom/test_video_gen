"""LUT Calibration Server using FlatBuffers for receiving NV12 frames."""

import socket
import flatbuffers
import numpy as np
import argparse
import os
from pathlib import Path
import json
import time
from datetime import datetime

# FlatBuffers imports
from hyperionnet.Request import Request
from hyperionnet.Image import Image
from hyperionnet.RawImage import RawImage
from hyperionnet.NV12Image import NV12Image
from hyperionnet.Register import Register
from hyperionnet.Reply import Reply, ReplyStart, ReplyEnd, ReplyAddVideo, ReplyAddRegistered

# Import the LUT Builder
from lut_builder import LutBuilder

# Import constants
from src.utils.constants import ChromaFormat, ColorRange


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LUT Calibration Server")
    parser.add_argument("--metadata", type=str, default="output/pattern_metadata.json",
                      help="Path to pattern metadata JSON file")
    parser.add_argument("--output-dir", type=str, default="output",
                      help="Output directory")
    parser.add_argument("--port", type=int, default=9999,
                      help="Server port")
    parser.add_argument("--save-frames", action="store_true",
                      help="Save received frames to disk")
    parser.add_argument("--chroma-format", type=str, default="422", choices=["420", "422", "444"],
                      help="Chroma subsampling format (420, 422, 444)")
    parser.add_argument("--color-range", type=str, default="limited", choices=["limited", "full"],
                      help="Color range (limited, full)")
    return parser.parse_args()


def start_server(args):
    """Start the LUT calibration server."""
    # Convert format and range strings to constants
    chroma_format = getattr(ChromaFormat, f"YUV_{args.chroma_format}")
    color_range = getattr(ColorRange, args.color_range.upper())
    
    print(f"Using format {chroma_format}, range {color_range}")
    
    # Initialize LUT Builder with specified format and range
    print("Initializing LUT Builder...")
    lut_builder = LutBuilder(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        chroma_format=chroma_format,
        color_range=color_range
    )
    
    # Create frames directory if saving frames
    frames_dir = None
    if args.save_frames:
        frames_dir = Path(args.output_dir) / "frames"
        frames_dir.mkdir(exist_ok=True, parents=True)
    
    # Create stats file for tracking
    stats_file = Path(args.output_dir) / "server_stats.json"
    stats = {
        "start_time": datetime.now().isoformat(),
        "frames_received": 0,
        "frames_processed": 0,
        "patterns_recognized": 0,
        "chroma_format": args.chroma_format,
        "color_range": args.color_range,
        "last_update": datetime.now().isoformat()
    }
    
    # Server setup
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = args.port
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")
    
    try:
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"[+] Accepted connection from {client_address}")
            
            process_client(client_socket, lut_builder, frames_dir, stats, stats_file)
    except KeyboardInterrupt:
        print("\n[*] Server stopped by user")
    finally:
        # Save final stats
        stats["end_time"] = datetime.now().isoformat()
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[*] Stats saved to {stats_file}")
        server_socket.close()


def process_client(client_socket, lut_builder, frames_dir, stats, stats_file):
    """Process a client connection."""
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
                process_request(request, client_socket, lut_builder, frames_dir, stats)
                
                # Update stats file periodically
                stats["last_update"] = datetime.now().isoformat()
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                    
            except Exception as e:
                print(f"[-] Error processing request: {e}")
                
    except Exception as e:
        print(f"[-] Error handling client: {e}")
    finally:
        client_socket.close()
        print(f"[*] Client connection closed")


def process_request(request, client_socket, lut_builder, frames_dir, stats):
    """Process a FlatBuffers request."""
    command_type = request.CommandType()
    print(f"[*] Command type: {command_type}")
    
    if command_type == 2:  # Image command
        process_image(request, client_socket, lut_builder, frames_dir, stats)
    elif command_type == 4:  # Registration
        process_registration(request, client_socket)
    else:
        print(f"[*] Unknown command type: {command_type}")


def process_image(request, client_socket, lut_builder, frames_dir, stats):
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
        
        # Raw image not supported for LUT building in this implementation
        print(f"[-] RawImage format not supported for LUT building")
        
        # Optionally save the frame
        if frames_dir:
            frame_path = frames_dir / f"raw_frame_{stats['frames_received']}.raw"
            with open(frame_path, "wb") as f:
                f.write(data.tobytes())
            print(f"[*] Saved raw frame to {frame_path}")
        
    elif image_type == 2:  # NV12Image
        nv12_image = NV12Image()
        nv12_image.Init(image.Data().Bytes, image.Data().Pos)
        width, height = nv12_image.Width(), nv12_image.Height()
        y_data = nv12_image.YDataAsNumpy()
        uv_data = nv12_image.UvDataAsNumpy()
        print(f"[*] Received NV12Image with dimensions {width}x{height}")
        
        # Get format and range from LUT builder
        chroma_format = lut_builder.chroma_format
        color_range = lut_builder.color_range
        print(f"[*] Processing with format {chroma_format}, range {color_range}")
        
        # Process frame for LUT building
        start_time = time.time()
        success = lut_builder.process_frame(y_data, uv_data, width, height)
        processing_time = time.time() - start_time
        
        if success:
            stats["frames_processed"] += 1
            stats["patterns_recognized"] += 1
            print(f"[*] Successfully processed frame for LUT in {processing_time:.2f}s")
        else:
            print(f"[*] Frame processing failed or skipped in {processing_time:.2f}s")
        
        # Optionally save the frame
        if frames_dir:
            # Include format and range in the filename
            frame_path = frames_dir / f"nv12_frame_{stats['frames_received']}_{chroma_format}_{color_range}.yuv"
            with open(frame_path, "wb") as f:
                f.write(y_data.tobytes())
                f.write(uv_data.tobytes())
            print(f"[*] Saved NV12 frame to {frame_path}")
    
    # Send response to prevent connection closure
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


if __name__ == "__main__":
    args = parse_args()
    start_server(args)