#!/usr/bin/env python3
"""
Clean up metadata.csv by removing entries for deleted audio files
Run this after you delete audio files to keep CSV in sync
"""

import pandas as pd
from pathlib import Path
import argparse
import shutil
from datetime import datetime


def clean_metadata_csv(base_dir: str, speaker_name: str, backup: bool = True):
    """
    Clean metadata.csv by removing entries for missing audio files
    
    Args:
        base_dir: Base output directory
        speaker_name: Speaker folder name
        backup: Create backup of original CSV (default: True)
    """
    
    base_path = Path(base_dir) / speaker_name
    metadata_path = base_path / "metadata.csv"
    audio_dir = base_path / "audio"
    
    print("="*70)
    print("METADATA.CSV CLEANUP TOOL")
    print("="*70)
    print(f"\nSpeaker: {speaker_name}")
    print(f"Metadata: {metadata_path}")
    print(f"Audio dir: {audio_dir}")
    
    # Check if files exist
    if not metadata_path.exists():
        print(f"\n❌ ERROR: metadata.csv not found at: {metadata_path}")
        return False
    
    if not audio_dir.exists():
        print(f"\n❌ ERROR: audio directory not found at: {audio_dir}")
        return False
    
    # Load CSV
    print(f"\n📂 Loading metadata.csv...")
    df = pd.read_csv(metadata_path)
    original_count = len(df)
    print(f"   ✓ Found {original_count} entries in CSV")
    
    # Check which files exist
    print(f"\n🔍 Checking audio files...")
    missing_files = []
    valid_rows = []
    
    for idx, row in df.iterrows():
        audio_file = audio_dir / row['file_name']
        if audio_file.exists():
            valid_rows.append(row)
        else:
            missing_files.append(row['file_name'])
    
    # Report results
    print(f"\n📊 Results:")
    print(f"   ✓ Found: {len(valid_rows)} audio files")
    print(f"   ❌ Missing: {len(missing_files)} audio files")
    
    if len(missing_files) == 0:
        print(f"\n✅ All files in CSV exist! No cleanup needed.")
        return True
    
    # Show missing files
    print(f"\n📝 Missing files that will be removed from CSV:")
    for f in missing_files[:20]:  # Show first 20
        print(f"   - {f}")
    if len(missing_files) > 20:
        print(f"   ... and {len(missing_files) - 20} more")
    
    # Confirm cleanup
    print(f"\n⚠️  This will:")
    print(f"   1. Remove {len(missing_files)} entries from metadata.csv")
    print(f"   2. Keep {len(valid_rows)} valid entries")
    if backup:
        print(f"   3. Create backup: metadata.csv.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    response = input(f"\n❓ Proceed with cleanup? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n❌ Cleanup cancelled.")
        return False
    
    # Create backup
    if backup:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = base_path / f"metadata.csv.backup.{timestamp}"
        shutil.copy2(metadata_path, backup_path)
        print(f"\n💾 Backup created: {backup_path.name}")
    
    # Create cleaned dataframe
    df_clean = pd.DataFrame(valid_rows)
    
    # Save cleaned CSV
    df_clean.to_csv(metadata_path, index=False)
    print(f"\n✅ Cleaned metadata.csv saved!")
    print(f"   Removed: {len(missing_files)} entries")
    print(f"   Remaining: {len(df_clean)} entries")
    
    # Summary
    print(f"\n{'='*70}")
    print("CLEANUP COMPLETE")
    print("="*70)
    print(f"Original entries: {original_count}")
    print(f"Removed entries:  {len(missing_files)}")
    print(f"Final entries:    {len(df_clean)}")
    print(f"Success rate:     {len(df_clean)/original_count*100:.1f}%")
    print("="*70 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Clean metadata.csv by removing entries for deleted audio files"
    )
    parser.add_argument(
        '--base-dir',
        required=True,
        help='Base output directory (e.g., C:/ChatterboxTraining/output)'
    )
    parser.add_argument(
        '--speaker',
        required=True,
        help='Speaker folder name (e.g., "00 Prolog")'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup of original CSV'
    )
    
    args = parser.parse_args()
    
    # FIX: Corrected Python syntax from args.base-dir to args.base_dir
    success = clean_metadata_csv(
        base_dir=args.base_dir,
        speaker_name=args.speaker,
        backup=not args.no_backup
    )
    
    if success:
        print("✓ You can now run load_dataset.bat with the cleaned CSV")
    else:
        print("✗ Cleanup failed")


if __name__ == "__main__":
    main()