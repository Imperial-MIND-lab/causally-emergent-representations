#!/usr/bin/env python3
import os
import requests
from pathlib import Path
import tqdm

ZENODO_URL = "https://zenodo.org/record/15555449/files"
DATASETS = {
    "fmri": {
        "filename": "Schaefer100_BOLD_HCP.mat",
        "dir": "fmri"
    },
    "meg": {
        "filename": "MEGdataset",
        "dir": "meg"
    },
    "ecog": [
        {"filename": f"ECoG_ch{i}.mat", "dir": "ecog"} 
        for i in range(1, 65)
    ]
}

def download_file(url: str, dest_path: Path, desc: str = None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(dest_path, 'wb') as f, tqdm.tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def main():
    # Create data directories
    data_dir = Path(__file__).parent
    for dataset in DATASETS.values():
        if isinstance(dataset, list):
            # Handle ECoG multiple files
            Path(data_dir / dataset[0]["dir"]).mkdir(parents=True, exist_ok=True)
        else:
            Path(data_dir / dataset["dir"]).mkdir(parents=True, exist_ok=True)
    
    # Download FMRI dataset
    fmri_path = data_dir / DATASETS["fmri"]["dir"] / DATASETS["fmri"]["filename"]
    if not fmri_path.exists():
        print(f"Downloading FMRI dataset...")
        download_file(
            f"{ZENODO_URL}/{DATASETS['fmri']['filename']}", 
            fmri_path,
            "Downloading FMRI dataset"
        )
    
    # Download MEG dataset
    meg_path = data_dir / DATASETS["meg"]["dir"] / DATASETS["meg"]["filename"]
    if not meg_path.exists():
        print(f"Downloading MEG dataset...")
        download_file(
            f"{ZENODO_URL}/{DATASETS['meg']['filename']}", 
            meg_path,
            "Downloading MEG dataset"
        )
    
    # Download ECoG datasets
    for ecog_file in DATASETS["ecog"]:
        ecog_path = data_dir / ecog_file["dir"] / ecog_file["filename"]
        if not ecog_path.exists():
            print(f"Downloading {ecog_file['filename']}...")
            download_file(
                f"{ZENODO_URL}/{ecog_file['filename']}", 
                ecog_path,
                f"Downloading {ecog_file['filename']}"
            )

if __name__ == "__main__":
    main() 