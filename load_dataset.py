#!/usr/bin/env python3
"""
Load your speech splitter dataset into HuggingFace format for Chatterbox training
Optimized for Windows + RTX 3060 6GB
HARDCODED FOR 24000Hz - REQUIRED BY BARK
"""

import pandas as pd
from datasets import Dataset, Audio, DatasetDict
from pathlib import Path
import argparse
import torchaudio
from functools import partial
import multiprocessing as mp
import numpy as np


def check_audio_exists(row, audio_dir):
    """Check if audio file exists (used for multiprocessing)"""
    audio_path = audio_dir / row['file_name']
    return audio_path.exists()


def load_and_resample_audio(example, target_sample_rate):
    """Load and resample audio using torchaudio"""
    path = example["audio_path"]
    try:
        audio_array, sr = torchaudio.load(path)
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            audio_array = resampler(audio_array)
        # Ensure we have a 1D array for mono audio
        if audio_array.dim() > 1:
            audio_array = audio_array.squeeze(0)  # Remove channel dim if present
        # Convert to numpy array explicitly
        audio_numpy = audio_array.numpy()
        example["audio"] = {
            "array": audio_numpy,
            "sampling_rate": target_sample_rate
        }
    except Exception as e:
        print(f"Error loading {path}: {e}")
        # Return empty array as fallback
        example["audio"] = {
            "array": np.array([], dtype=np.float32),
            "sampling_rate": target_sample_rate
        }
    return example


def load_speech_splitter_dataset(
    base_dir: str,
    speaker_name: str,
    train_split: float = 0.9,
    target_sample_rate: int = 24000,  # <-- FIX: Hardcoded to 24000Hz
    num_proc: int = None
):
    """
    Load dataset from your speech_splitter_v2.py output format
    """
    base_path = Path(base_dir) / speaker_name
    metadata_path = base_path / "metadata.csv"
    audio_dir = base_path / "audio"

    print(f"Loading dataset from: {base_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Audio directory: {audio_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio directory not found at: {audio_dir}")

    print("\nReading metadata.csv...")
    df = pd.read_csv(metadata_path)
    print(f"✓ Found {len(df)} samples")

    if 'file_name' not in df.columns or 'text' not in df.columns:
        raise ValueError(f"Expected columns 'file_name' and 'text', got: {df.columns.tolist()}")

    # Add full audio paths
    df['audio_path'] = df['file_name'].apply(lambda x: str(audio_dir / x))

    print("\nChecking audio files (multiprocessing)...")
    with mp.Pool(processes=num_proc or mp.cpu_count()) as pool:
        exists = pool.map(
            partial(check_audio_exists, audio_dir=audio_dir),
            df.to_dict('records')
        )

    valid_df = df[exists].reset_index(drop=True)
    missing_count = len(df) - len(valid_df)

    if missing_count > 0:
        print(f"\n⚠ WARNING: {missing_count} audio files missing. Removing from dataset.")
    print(f"✓ Kept {len(valid_df)} valid samples.")

    if len(valid_df) == 0:
        raise ValueError("No valid audio files found!")

    # Create HF Dataset
    dataset = Dataset.from_pandas(valid_df[['audio_path', 'text', 'file_name']])

    print(f"\nLoading and resampling audio to {target_sample_rate}Hz (multiprocessing)...")
    dataset = dataset.map(
        partial(load_and_resample_audio, target_sample_rate=target_sample_rate),
        num_proc=num_proc or mp.cpu_count()
    )

    # Remove raw path column
    dataset = dataset.remove_columns(["audio_path"])

    # Split into train/validation
    print(f"\nSplitting dataset: {train_split*100:.0f}% train, {(1-train_split)*100:.0f}% validation")
    split_dataset = dataset.train_test_split(
        test_size=1-train_split,
        seed=42,
        shuffle=True
    )

    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

    print(f"\n{'='*60}")
    print(f"DATASET LOADED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")
    print(f"Sample rate: {target_sample_rate}Hz")  # <-- FIX: Will now show 24000Hz
    print(f"Language: Czech (cs)")
    print(f"{'='*60}\n")

    # Show a sample
    print("Sample data:")
    sample = dataset_dict['train'][0]
    audio_data = sample['audio']
    audio_array = audio_data['array']
    
    # Handle case where audio might be a list or have unexpected type
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array)
    
    if hasattr(audio_array, 'shape'):
        shape_str = str(audio_array.shape)
    else:
        shape_str = f"Type: {type(audio_array)}, Length: {len(audio_array) if hasattr(audio_array, '__len__') else 'N/A'}"
    
    print(f"  Audio shape: {shape_str}")
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  File: {sample['file_name']}")

    return dataset_dict


def save_dataset(dataset_dict: DatasetDict, output_path: str):
    """Save processed dataset to disk"""
    print(f"\nSaving dataset to: {output_path}")
    dataset_dict.save_to_disk(output_path)
    print("✓ Dataset saved!")


def main():
    parser = argparse.ArgumentParser(
        description="Load speech_splitter_v2.py output into HuggingFace format"
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
        '--output',
        default='./processed_dataset',
        help='Output path for processed dataset'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
        help='Train/validation split ratio (default: 0.9)'
    )
    # --- FIX: Removed --sample-rate argument ---
    parser.add_argument(
        '--num-proc',
        type=int,
        default=None,
        help='Number of processes to use (default: CPU count)'
    )

    args = parser.parse_args()
    
    # --- FIX: Hardcode target_sample_rate to 24000Hz ---
    target_sample_rate = 24000

    try:
        dataset_dict = load_speech_splitter_dataset(
            base_dir=args.base_dir,
            speaker_name=args.speaker,
            train_split=args.train_split,
            target_sample_rate=target_sample_rate,
            num_proc=args.num_proc
        )

        save_dataset(dataset_dict, args.output)

        print("\n✓ Done! You can now use this dataset for training.")
        print(f"  Load it with: dataset = load_from_disk('{args.output}')")
        
    except Exception as e:
        print(f"\nERROR: Dataset loading failed!")
        print(f"Check the error message above.")
        raise e


if __name__ == "__main__":
    main()