#!/usr/bin/env python3
"""
GPU Form Factor Detection Script
Determines if NVIDIA GPUs are PCIe or SXM form factor
"""

import subprocess
import sys
import re
from typing import List, Dict, Optional

def run_command(cmd: str) -> tuple[str, int]:
    """Run a shell command and return output and exit code"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return f"Error: {e}", 1

def check_nvidia_driver() -> bool:
    """Check if NVIDIA driver is installed"""
    output, code = run_command("nvidia-smi --version")
    return code == 0

def get_gpu_list() -> List[Dict]:
    """Get list of GPUs from nvidia-smi"""
    output, code = run_command("nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader,nounits")
    if code != 0:
        return []
    
    gpus = []
    for line in output.split('\n'):
        if line.strip():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                gpus.append({
                    'index': parts[0],
                    'name': parts[1],
                    'pci_bus_id': parts[2]
                })
    return gpus

def check_pcie_enumeration() -> List[str]:
    """Check which NVIDIA devices appear in PCIe enumeration"""
    output, code = run_command("lspci | grep -i nvidia")
    if code != 0:
        return []
    
    pcie_devices = []
    for line in output.split('\n'):
        if line.strip():
            # Extract bus ID (e.g., "03:00.0")
            match = re.match(r'^([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])', line)
            if match:
                pcie_devices.append(match.group(1))
    return pcie_devices

def get_detailed_pcie_info(bus_id: str) -> Dict:
    """Get detailed PCIe information for a specific device"""
    output, code = run_command(f"lspci -s {bus_id} -v")
    if code != 0:
        return {}
    
    info = {}
    for line in output.split('\n'):
        line = line.strip()
        if 'Subsystem:' in line:
            info['subsystem'] = line.split('Subsystem:')[1].strip()
        elif 'Physical Slot:' in line:
            info['physical_slot'] = line.split('Physical Slot:')[1].strip()
        elif 'LnkCap:' in line:
            info['link_capabilities'] = line
        elif 'LnkSta:' in line:
            info['link_status'] = line
    
    return info

def determine_form_factor(gpu: Dict, pcie_devices: List[str]) -> str:
    """Determine if a GPU is PCIe or SXM based on available information"""
    pci_bus_id = gpu['pci_bus_id']
    
    # Extract short bus ID (remove domain if present)
    short_bus_id = pci_bus_id.split(':')[-2] + ':' + pci_bus_id.split(':')[-1]
    
    # Check if GPU appears in PCIe enumeration
    pcie_found = any(short_bus_id in device for device in pcie_devices)
    
    if pcie_found:
        return "PCIe"
    else:
        # Double-check with direct lspci query
        output, code = run_command(f"lspci -s {short_bus_id}")
        if code == 0 and output.strip():
            return "PCIe"
        else:
            return "SXM"

def get_gpu_specs(gpu_index: str) -> Dict:
    """Get additional GPU specifications"""
    specs = {}
    
    # Get power limit
    output, code = run_command(f"nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i {gpu_index}")
    if code == 0:
        specs['power_limit'] = output.strip() + "W"
    
    # Get memory info
    output, code = run_command(f"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i {gpu_index}")
    if code == 0:
        memory_mb = float(output.strip())
        specs['memory'] = f"{memory_mb/1024:.0f}GB"
    
    return specs

def main():
    print("🔍 GPU Form Factor Detection Script")
    print("=" * 50)
    
    # Check if NVIDIA driver is available
    if not check_nvidia_driver():
        print("❌ NVIDIA driver not found or nvidia-smi not available")
        sys.exit(1)
    
    # Get GPU list
    gpus = get_gpu_list()
    if not gpus:
        print("❌ No NVIDIA GPUs found")
        sys.exit(1)
    
    print(f"📊 Found {len(gpus)} NVIDIA GPU(s)")
    print()
    
    # Check PCIe enumeration
    pcie_devices = check_pcie_enumeration()
    print(f"🔌 PCIe enumerated NVIDIA devices: {len(pcie_devices)}")
    for device in pcie_devices:
        print(f"   - {device}")
    print()
    
    # Analyze each GPU
    for i, gpu in enumerate(gpus):
        print(f"🎯 GPU {gpu['index']}: {gpu['name']}")
        print(f"   PCI Bus ID: {gpu['pci_bus_id']}")
        
        # Determine form factor
        form_factor = determine_form_factor(gpu, pcie_devices)
        
        # Get additional specs
        specs = get_gpu_specs(gpu['index'])
        
        # Get detailed PCIe info if it's PCIe
        if form_factor == "PCIe":
            short_bus_id = gpu['pci_bus_id'].split(':')[-2] + ':' + gpu['pci_bus_id'].split(':')[-1]
            pcie_info = get_detailed_pcie_info(short_bus_id)
            
            print(f"   📋 Form Factor: ✅ {form_factor}")
            if 'power_limit' in specs:
                print(f"   ⚡ Power Limit: {specs['power_limit']}")
            if 'memory' in specs:
                print(f"   💾 Memory: {specs['memory']}")
            
            if pcie_info:
                if 'physical_slot' in pcie_info:
                    print(f"   🔌 Physical Slot: {pcie_info['physical_slot']}")
                if 'link_status' in pcie_info:
                    # Extract speed and width from link status
                    link_match = re.search(r'Speed ([^,]+).*Width ([^,]+)', pcie_info['link_status'])
                    if link_match:
                        speed, width = link_match.groups()
                        print(f"   🚀 PCIe Link: {speed.strip()} {width.strip()}")
        else:
            print(f"   📋 Form Factor: 🔥 {form_factor}")
            if 'power_limit' in specs:
                print(f"   ⚡ Power Limit: {specs['power_limit']}")
            if 'memory' in specs:
                print(f"   💾 Memory: {specs['memory']}")
            print(f"   🔌 Connection: SXM5 connector (not PCIe)")
        
        print()
    
    # Summary
    pcie_count = sum(1 for gpu in gpus if determine_form_factor(gpu, pcie_devices) == "PCIe")
    sxm_count = len(gpus) - pcie_count
    
    print("📈 Summary:")
    print(f"   PCIe GPUs: {pcie_count}")
    print(f"   SXM GPUs: {sxm_count}")
    
    if pcie_count > 0 and sxm_count > 0:
        print("   🔀 Mixed configuration detected!")
    elif pcie_count > 0:
        print("   ✅ All GPUs are PCIe form factor")
    else:
        print("   🔥 All GPUs are SXM form factor")

if __name__ == "__main__":
    main()
