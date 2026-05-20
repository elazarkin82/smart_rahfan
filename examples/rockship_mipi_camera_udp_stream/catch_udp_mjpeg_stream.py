#!/usr/bin/env python3
import sys
import socket
import threading
import time
import io
import tkinter as tk
from tkinter import messagebox

# Attempt to import Pillow library for image processing
try:
    from PIL import Image, ImageTk
except ImportError:
    print("\n[ERROR] Pillow (PIL) library is required to decode JPEG stream.")
    print("Please install it on your system using:")
    print("    sudo apt-get update && sudo apt-get install python3-pil python3-pil.imagetk")
    print("Or:")
    print("    pip3 install Pillow\n")
    sys.exit(1)

class MjpegReceiverApp:
    def __init__(self, root, port):
        self.root = root
        self.port = port
        self.running = True
        
        # Configure Premium Dark Mode Styling
        self.bg_color = "#121212"
        self.card_color = "#1E1E1E"
        self.accent_color = "#00E676"  # Vibrant Neon Green
        self.text_color = "#E0E0E0"
        self.dim_text_color = "#888888"
        
        self.root.title("Smart Rahfan - UDP MJPEG Receiver")
        self.root.geometry("640x580")
        self.root.configure(bg=self.bg_color)
        
        # Header Panel
        self.header_frame = tk.Frame(self.root, bg=self.bg_color, pady=10)
        self.header_frame.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            self.header_frame, 
            text="SMART RAHFAN STREAMER", 
            font=("Helvetica", 16, "bold"), 
            fg=self.accent_color, 
            bg=self.bg_color
        )
        self.title_label.pack()
        
        self.subtitle_label = tk.Label(
            self.header_frame, 
            text=f"Listening on UDP Port {self.port}", 
            font=("Helvetica", 10), 
            fg=self.dim_text_color, 
            bg=self.bg_color
        )
        self.subtitle_label.pack()

        # Stream Display Panel (Canvas)
        # Default placeholder dimensions: 320x240 (QVGA)
        self.canvas_width = 480
        self.canvas_height = 360
        self.canvas_frame = tk.Frame(self.root, bg=self.bg_color, bd=2, highlightthickness=1, highlightbackground="#333333")
        self.canvas_frame.pack(pady=10)
        
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg="#0A0A0A", 
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Place centered text placeholder inside canvas
        self.placeholder_text = self.canvas.create_text(
            self.canvas_width // 2, 
            self.canvas_height // 2, 
            text="Awaiting Video Stream...", 
            fill=self.dim_text_color, 
            font=("Helvetica", 12, "italic")
        )

        # Status and Telemetry Bar
        self.status_frame = tk.Frame(self.root, bg=self.card_color, padx=15, pady=12, bd=0)
        self.status_frame.pack(fill=tk.X, padx=20, pady=10, side=tk.BOTTOM)
        
        # Grid layout for structured stats
        self.status_frame.columnconfigure(0, weight=1)
        self.status_frame.columnconfigure(1, weight=1)
        self.status_frame.columnconfigure(2, weight=1)
        
        # FPS Counter Display
        self.fps_label = tk.Label(
            self.status_frame, 
            text="FPS: 0.0", 
            font=("Helvetica", 12, "bold"), 
            fg=self.accent_color, 
            bg=self.card_color
        )
        self.fps_label.grid(row=0, column=0, sticky="w")
        
        # Frame Size Display
        self.size_label = tk.Label(
            self.status_frame, 
            text="Size: 0.0 KB", 
            font=("Helvetica", 11), 
            fg=self.text_color, 
            bg=self.card_color
        )
        self.size_label.grid(row=0, column=1)
        
        # Stream Status Info
        self.state_label = tk.Label(
            self.status_frame, 
            text="Status: DISCONNECTED", 
            font=("Helvetica", 11), 
            fg="#FF1744",  # Neon Red
            bg=self.card_color
        )
        self.state_label.grid(row=0, column=2, sticky="e")

        # Stats variables
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.photo = None  # To retain image reference and prevent garbage collection
        
        # Handle Window Close gracefully
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start background receiver thread
        self.receiver_thread = threading.Thread(target=self.receive_udp_stream, daemon=True)
        self.receiver_thread.start()

    def receive_udp_stream(self):
        # Create and bind UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.port))
            sock.settimeout(0.5)  # Clean check loop exit
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Socket Error", f"Failed to bind to port {self.port}:\n{e}"))
            self.running = False
            self.root.after(0, self.root.destroy)
            return

        fps_timer = time.time()
        
        while self.running:
            try:
                # Max UDP size is 65535 bytes
                data, addr = sock.recvfrom(65536)
            except socket.timeout:
                # No data received within timeout, set status to WAITING
                self.root.after(0, lambda: self.state_label.configure(text="Status: AWAITING DATA", fg="#FFC107")) # Amber
                continue
            except Exception as e:
                if self.running:
                    print(f"[ERROR] Socket error: {e}")
                break

            if not data:
                continue

            frame_size_kb = len(data) / 1024.0
            
            # FPS Calculation
            self.frame_count += 1
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                fps_timer = now
            
            try:
                # Open JPEG data from bytes memory stream
                image_file = io.BytesIO(data)
                img = Image.open(image_file)
                
                # Dynamic scaling to fit tkinter canvas elegantly
                img.thumbnail((self.canvas_width, self.canvas_height))
                
                # Convert PIL Image to Tkinter PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Safely update GUI in the main Tkinter thread
                self.root.after(0, self.update_gui_frame, photo, frame_size_kb, self.fps)
                
            except Exception as e:
                # Handle corrupted/incomplete packet frames safely
                print(f"[WARNING] Failed to decode received frame: {e}")

        sock.close()

    def update_gui_frame(self, photo, size_kb, fps):
        # Update image on canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas_width // 2, 
            self.canvas_height // 2, 
            image=photo, 
            anchor=tk.CENTER
        )
        self.photo = photo  # Keep reference to prevent GC garbage collection
        
        # Update status labels
        self.fps_label.configure(text=f"FPS: {fps:.1f}")
        self.size_label.configure(text=f"Size: {size_kb:.1f} KB")
        self.state_label.configure(text="Status: RECEIVING", fg=self.accent_color)

    def on_close(self):
        self.running = False
        self.root.destroy()

def main():
    # Require port argument
    if len(sys.argv) < 2:
        print("\n[ERROR] Port argument is required.")
        print("Usage:")
        print("    python3 catch_udp_mjpeg_stream.py <PORT>")
        print("Example:")
        print("    python3 catch_udp_mjpeg_stream.py 5000\n")
        sys.exit(1)
        
    try:
        port = int(sys.argv[1])
        if port < 1024 or port > 65535:
            raise ValueError("Port must be between 1024 and 65535.")
    except ValueError as e:
        print(f"\n[ERROR] Invalid port number: {e}\n")
        sys.exit(1)
        
    root = tk.Tk()
    app = MjpegReceiverApp(root, port)
    root.mainloop()

if __name__ == "__main__":
    main()
