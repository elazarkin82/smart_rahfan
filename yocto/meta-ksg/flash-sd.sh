#!/usr/bin/env python3
import os
import sys
import glob
import subprocess

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Define the target USB Port/ID to flash to (e.g. "1-1.3", "2-3").
# Locked to "2-3" as confirmed by the user.
TARGET_USB_PORT = "2-3" 

# Path to the build deploy images directory
DEPLOY_IMAGES_DIR = "build/tmp/deploy/images/radxa-zero-3e"
# ==============================================================================

def read_sysfs_file(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except Exception:
        return ""

def get_usb_devices():
    devices = []
    base_dir = "/sys/bus/usb/devices"
    if not os.path.exists(base_dir):
        print("Error: /sys/bus/usb/devices not found.")
        return devices

    for name in os.listdir(base_dir):
        # Skip interfaces (e.g. 1-1:1.0) and virtual/root hubs (like usb1, usb2)
        if ":" in name or name.startswith("usb"):
            continue

        device_path = os.path.join(base_dir, name)
        if not os.path.isdir(device_path):
            continue

        vendor = read_sysfs_file(os.path.join(device_path, "idVendor"))
        product_id = read_sysfs_file(os.path.join(device_path, "idProduct"))
        manufacturer = read_sysfs_file(os.path.join(device_path, "manufacturer"))
        product_name = read_sysfs_file(os.path.join(device_path, "product"))

        # Find block devices associated with this USB device
        block_devs = []
        for root, dirs, files in os.walk(device_path):
            if 'block' in dirs:
                block_dir = os.path.join(root, 'block')
                try:
                    block_devs.extend(os.listdir(block_dir))
                except Exception:
                    pass

        # Also search for block device in parent-child hierarchy in case it is deeply nested
        if not block_devs:
            for root, dirs, files in os.walk(device_path):
                for d in dirs:
                    if d.startswith("sd") and not any(c.isdigit() for c in d):
                        block_devs.append(d)
            block_devs = list(set(block_devs))

        devices.append({
            "port": name,
            "vendor": vendor,
            "product_id": product_id,
            "manufacturer": manufacturer,
            "product_name": product_name,
            "block_devices": block_devs
        })
    return devices

def find_image_file():
    if not os.path.exists(DEPLOY_IMAGES_DIR):
        return None, None

    # Look for .wic files in the deploy directory
    # We prefer the generic core-image-minimal symlink if it exists, or any .wic file
    patterns = [
        os.path.join(DEPLOY_IMAGES_DIR, "core-image-minimal-radxa-zero-3e.wic"),
        os.path.join(DEPLOY_IMAGES_DIR, "*.wic"),
        os.path.join(DEPLOY_IMAGES_DIR, "*.wic.gz"),
        os.path.join(DEPLOY_IMAGES_DIR, "*.wic.bz2")
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            # Sort to get the most specific one or resolve symlink
            target_file = sorted(files)[0]
            # Look for a corresponding .bmap file
            bmap_file = target_file + ".bmap"
            if not os.path.exists(bmap_file):
                # Try replacing extension
                base_without_ext = os.path.splitext(target_file)[0]
                possible_bmap = base_without_ext + ".bmap"
                bmap_file = possible_bmap if os.path.exists(possible_bmap) else None
            else:
                bmap_file = bmap_file if os.path.exists(bmap_file) else None
            return target_file, bmap_file

    return None, None

def is_tool_installed(name):
    from shutil import which
    return which(name) is not None

def unmount_target(dev_name):
    print(f"[INFO] Checking for mounted partitions on /dev/{dev_name}...")
    try:
        with open("/proc/mounts", "r") as f:
            mounts = []
            for line in f:
                parts = line.split()
                if parts and parts[0].startswith(f"/dev/{dev_name}"):
                    mounts.append(parts[0])
        
        if not mounts:
            print("[INFO] No partitions are currently mounted.")
            return True

        for mount in mounts:
            print(f"[INFO] Unmounting {mount}...")
            # Run sudo umount
            res = subprocess.run(["sudo", "umount", mount], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[WARNING] Failed to unmount {mount}: {res.stderr.strip()}")
                return False
        return True
    except Exception as e:
        print(f"[WARNING] Error checking/unmounting partitions: {e}")
        return False

def main():
    print("=== USB Devices Scan ===")
    devices = get_usb_devices()
    
    # 1. Print all existing USB devices and what is connected to them
    for dev in sorted(devices, key=lambda x: x["port"]):
        dev_str = f"Port: {dev['port']}"
        if dev['vendor'] or dev['product_id']:
            dev_str += f" | ID: {dev['vendor']}:{dev['product_id']}"
        if dev['manufacturer'] or dev['product_name']:
            name_parts = [dev['manufacturer'], dev['product_name']]
            name = " ".join([p for p in name_parts if p])
            dev_str += f" | Name: {name}"
        
        if dev['block_devices']:
            dev_str += f" | Storage: {', '.join(['/dev/' + b for b in dev['block_devices']])}"
        else:
            dev_str += " | [No Storage]"
        print(dev_str)

    print("\n=== Target USB Configuration ===")
    if not TARGET_USB_PORT:
        print("[INFO] TARGET_USB_PORT is not defined. Please set it in the script configuration section.")
        print("[INFO] Safety Mode Active: No flashing target specified. Exiting safely.")
        sys.exit(0)

    # 2. Check if the specified TARGET_USB_PORT is connected
    target_dev = None
    for dev in devices:
        if dev["port"] == TARGET_USB_PORT:
            target_dev = dev
            break

    if not target_dev:
        print(f"[ERROR] Target USB Port '{TARGET_USB_PORT}' is NOT connected.")
        sys.exit(1)

    print(f"[FOUND] Target port '{TARGET_USB_PORT}' is connected.")
    dev_name_str = "Unknown Device"
    if target_dev['manufacturer'] or target_dev['product_name']:
        dev_name_str = f"{target_dev['manufacturer']} {target_dev['product_name']}".strip()
        print(f"        Device: {dev_name_str}")
    
    # Check if it is a storage device
    if not target_dev['block_devices']:
        print(f"[ERROR] Device at port '{TARGET_USB_PORT}' is NOT a storage device (no block devices found).")
        sys.exit(1)

    target_blocks = sorted(list(set(target_dev['block_devices'])))
    target_block = target_blocks[0]
    target_dev_path = f"/dev/{target_block}"
    
    print(f"[SUCCESS] Target port '{TARGET_USB_PORT}' maps to storage device: {target_dev_path}")

    # 3. Locate the WIC image
    print("\n=== Locating Yocto Image ===")
    wic_file, bmap_file = find_image_file()
    
    if not wic_file:
        print(f"[ERROR] No .wic image found in '{DEPLOY_IMAGES_DIR}'.")
        print("[ERROR] Please build the Yocto image first by running:")
        print("        source poky/oe-init-build-env build")
        print("        bitbake core-image-minimal")
        sys.exit(1)

    print(f"[FOUND] Image file: {wic_file}")
    if bmap_file:
        print(f"[FOUND] Bmap file:  {bmap_file}")
    else:
        print("[INFO]  Bmap file not found. Flashing will fall back to 'dd'.")

    # 4. User Confirmation Prompt (CRITICAL SAFETY GUARD)
    print("\n" + "!" * 80)
    print("WARNING: YOU ARE ABOUT TO FLASH A SYSTEM IMAGE TO A PHYSICAL DRIVE!")
    print(f"TARGET PORT:   {TARGET_USB_PORT}")
    print(f"TARGET DEVICE: {target_dev_path} ({dev_name_str})")
    print(f"SOURCE IMAGE:  {wic_file}")
    print("!" * 80)
    print("THIS ACTION WILL COMPLETELY WIPE ALL DATA ON THE TARGET DRIVE.")
    print("ALL OTHER DRIVES ARE SECURELY LOCKED AND PROTECTED.")
    print("!" * 80)
    
    try:
        confirm = input(f"Are you absolutely sure you want to write to {target_dev_path}? (type 'yes' to proceed): ")
    except KeyboardInterrupt:
        print("\n[INFO] Flashing cancelled by user.")
        sys.exit(0)

    if confirm.strip().lower() != "yes":
        print("[INFO] Confirmation failed. Flashing aborted.")
        sys.exit(0)

    # 5. Unmount partitions
    if not unmount_target(target_block):
        print("[ERROR] Failed to unmount partitions on the target drive. Aborting for safety.")
        sys.exit(1)

    # 6. Execute Flashing
    print("\n=== Flashing Storage ===")
    
    # Try using bmaptool if bmap file exists and tool is installed
    if bmap_file and is_tool_installed("bmaptool"):
        print(f"[RUNNING] Flashing via bmaptool (Fast Mode)...")
        cmd = ["sudo", "bmaptool", "copy", wic_file, target_dev_path]
        print(f"Command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("\n[SUCCESS] Flashing completed successfully via bmaptool!")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] bmaptool flashing failed: {e}")
            sys.exit(1)
    else:
        # Fallback to dd
        if bmap_file and not is_tool_installed("bmaptool"):
            print("[INFO] bmaptool is not installed. Falling back to dd.")
        
        print(f"[RUNNING] Flashing via dd (Standard Mode)...")
        # Command: sudo dd if=<wic_file> of=/dev/sdX bs=4M status=progress conv=fsync
        cmd = ["sudo", "dd", f"if={wic_file}", f"of={target_dev_path}", "bs=4M", "status=progress", "conv=fsync"]
        print(f"Command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("\n[SUCCESS] Flashing completed successfully via dd!")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] dd flashing failed: {e}")
            sys.exit(1)

    print("\n[INFO] You can now safely eject your SD card/USB drive.")

if __name__ == "__main__":
    main()
