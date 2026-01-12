"""
Script to clip audio files from 30s to 60s (30 second clips).
The clipped audio replaces the original files in-place.
Uses ffmpeg for audio processing.
"""

import os
import subprocess
import sys
from pathlib import Path

def clip_audio_file(input_path: Path, start_time: int = 30, duration: int = 30) -> bool:
    """
    Clip an audio file from start_time to start_time + duration.
    
    Args:
        input_path: Path to the audio file
        start_time: Start time in seconds (default: 30)
        duration: Duration in seconds (default: 30)
    
    Returns:
        True if successful, False otherwise
    """
    # Create a temporary output file
    temp_output = input_path.with_suffix('.temp.mp3')
    
    try:
        # Build ffmpeg command
        # -y: overwrite output file without asking
        # -ss: start time
        # -t: duration
        # -i: input file
        # -c copy: copy codec (no re-encoding, faster)
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', str(input_path),
            '-c', 'copy',
            str(temp_output)
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  Error clipping {input_path.name}: {result.stderr}")
            # Clean up temp file if it exists
            if temp_output.exists():
                temp_output.unlink()
            return False
        
        # Replace original file with clipped version
        input_path.unlink()  # Delete original
        temp_output.rename(input_path)  # Rename temp to original
        
        return True
        
    except Exception as e:
        print(f"  Exception clipping {input_path.name}: {e}")
        # Clean up temp file if it exists
        if temp_output.exists():
            temp_output.unlink()
        return False

def get_audio_duration(file_path: Path) -> float:
    """Get the duration of an audio file in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return 0.0

def process_directory(directory: Path) -> tuple[int, int, int]:
    """
    Process all audio files in a directory.
    
    Args:
        directory: Path to the directory containing audio files
    
    Returns:
        Tuple of (processed_count, skipped_count, failed_count)
    """
    processed = 0
    skipped = 0
    failed = 0
    
    # Get all mp3 files
    audio_files = list(directory.glob('*.mp3'))
    total = len(audio_files)
    
    print(f"\nProcessing {total} files in {directory.name}/")
    print("-" * 50)
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{total}] {audio_file.name[:50]}...", end=' ')
        
        # Check duration first
        duration = get_audio_duration(audio_file)
        
        if duration < 60:
            print(f"SKIPPED (duration: {duration:.1f}s < 60s)")
            skipped += 1
            continue
        
        if clip_audio_file(audio_file):
            print("OK")
            processed += 1
        else:
            print("FAILED")
            failed += 1
    
    return processed, skipped, failed

def main():
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Define directories to process
    directories = [
        script_dir / 'bangla',
        script_dir / 'english'
    ]
    
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    
    print("=" * 60)
    print("Audio Clipping Script")
    print("Clipping from 30s to 60s (30 second clips)")
    print("=" * 60)
    
    for directory in directories:
        if not directory.exists():
            print(f"\nDirectory not found: {directory}")
            continue
        
        processed, skipped, failed = process_directory(directory)
        total_processed += processed
        total_skipped += skipped
        total_failed += failed
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped (too short): {total_skipped}")
    print(f"  Failed: {total_failed}")
    print("=" * 60)

if __name__ == '__main__':
    main()
