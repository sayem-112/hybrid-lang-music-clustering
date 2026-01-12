"""
Script to combine Bangla and English song CSV files into a single hybrid_songs.csv file.
"""

import pandas as pd
from pathlib import Path

def main():
    # Define file paths
    script_dir = Path(__file__).parent
    bangla_file = script_dir / 'updated_bangla_songs_subset_250.csv'
    english_file = script_dir / 'updated_english_songs_subset_200.csv'
    output_file = script_dir / 'hybrid_songs.csv'
    
    print("Loading CSV files...")
    
    # Read both CSV files
    bangla_df = pd.read_csv(bangla_file)
    english_df = pd.read_csv(english_file)
    
    print(f"Bangla songs: {len(bangla_df)} rows")
    print(f"Bangla columns: {list(bangla_df.columns)}")
    print(f"English songs: {len(english_df)} rows")
    print(f"English columns: {list(english_df.columns)}")
    
    # Add language column to identify the source
    bangla_df['language'] = 'bangla'
    english_df['language'] = 'english'
    
    # Reorder columns to have consistent structure
    # Target order: song, artist, genre, lyrics, audio_path, language
    column_order = ['song', 'artist', 'genre', 'lyrics', 'audio_path', 'language']
    
    # Ensure both dataframes have the same column order
    bangla_df = bangla_df[column_order]
    english_df = english_df[column_order]
    
    # Combine the dataframes
    print("\nCombining dataframes...")
    hybrid_df = pd.concat([bangla_df, english_df], ignore_index=True)
    
    # Save to new file
    print(f"\nSaving to {output_file}...")
    hybrid_df.to_csv(output_file, index=False)
    
    print(f"\nDone! Created hybrid_songs.csv with {len(hybrid_df)} total songs.")
    print(f"  - Bangla songs: {len(bangla_df)}")
    print(f"  - English songs: {len(english_df)}")

if __name__ == '__main__':
    main()
