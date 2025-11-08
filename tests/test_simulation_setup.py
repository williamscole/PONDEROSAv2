#!/usr/bin/env python
"""
Interactive test script for simulation_workspace functionality.
Run this to manually test the workspace setup.
"""

from pathlib import Path
from ponderosa.simulation.config import SimulationConfig
from ponderosa.simulation.setup import simulation_workspace

def test_with_cleanup():
    """Test with automatic cleanup (default behavior)."""
    print("=" * 60)
    print("TEST 1: With Cleanup (cleanup_temp=True)")
    print("=" * 60)
    
    # Load config
    config_file = Path("tests/data/simulation2/args.yaml")
    config = SimulationConfig.from_yaml(config_file)
    config.cleanup_temp = True
    
    temp_path_reference = None
    
    with simulation_workspace(config) as temp_dir:
        temp_path_reference = temp_dir
        print(f"\n✓ Workspace created at: {temp_dir}")
        
        # Check symlinked files
        print("\n📁 Files in workspace:")
        for item in temp_dir.iterdir():
            if item.is_symlink():
                target = item.resolve()
                print(f"  {item.name} -> {target}")
            else:
                print(f"  {item.name}")
        
        # Verify key files exist
        vcf_file = temp_dir / "input.vcf.gz"
        if vcf_file.exists():
            print(f"\n✓ VCF file exists: {vcf_file}")
        else:
            print(f"\n✗ VCF file missing: {vcf_file}")
        
        input("Press Enter to exit the workspace (will trigger cleanup)...")
    
    # After exiting context
    print(f"\n🧹 Exited workspace context")
    print(f"Checking if temp directory still exists: {temp_path_reference}")
    if temp_path_reference.exists():
        print("✗ Directory still exists (cleanup failed!)")
    else:
        print("✓ Directory cleaned up successfully!")


def test_without_cleanup():
    """Test with preserved temp directory (cleanup_temp=False)."""
    print("\n" + "=" * 60)
    print("TEST 2: Without Cleanup (cleanup_temp=False)")
    print("=" * 60)
    
    # Load config
    config_file = Path("tests/data/simulation1/args.yaml")
    config = SimulationConfig.from_yaml(config_file)
    config.cleanup_temp = False
    
    temp_path_reference = None
    
    with simulation_workspace(config) as temp_dir:
        temp_path_reference = temp_dir
        print(f"\n✓ Workspace created at: {temp_dir}")
        
        # Check symlinked files
        print("\n📁 Files in workspace:")
        for item in temp_dir.iterdir():
            if item.is_symlink():
                target = item.resolve()
                print(f"  {item.name} -> {target}")
            else:
                print(f"  {item.name}")
    
    # After exiting context
    print(f"\n📦 Exited workspace context")
    print(f"Checking if temp directory still exists: {temp_path_reference}")
    if temp_path_reference.exists():
        print("✓ Directory preserved as expected!")
        print(f"\n💡 Remember to manually delete: rm -rf {temp_path_reference}")
    else:
        print("✗ Directory was cleaned up (unexpected!)")


def test_file_validation():
    """Test that missing files are caught."""
    print("\n" + "=" * 60)
    print("TEST 3: File Validation")
    print("=" * 60)
    
    # Load config
    config_file = Path("tests/data/simulation1/args.yaml")
    config = SimulationConfig.from_yaml(config_file)
    config.cleanup_temp = True
    
    # Test with valid files
    try:
        with simulation_workspace(config) as temp_dir:
            print(f"✓ Validation passed with valid config")
    except FileNotFoundError as e:
        print(f"✗ Unexpected error: {e}")


if __name__ == "__main__":
    print("SIMULATION WORKSPACE TESTING")
    print("=" * 60)
    
    # Run tests
    try:
        test_with_cleanup()
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
    
    try:
        test_without_cleanup()
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
    
    try:
        test_file_validation()
    except Exception as e:
        print(f"\n✗ Test 3 failed: {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)