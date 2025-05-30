# Dataset Setup Instructions

This directory contains the datasets used in the paper "Learning diverse causally emergent representations from time series data". The datasets are:

1. FMRI data (`fmri/Schaefer100_BOLD_HCP.mat`)
2. MEG data (`meg/MEGdataset.mat`)
3. ECoG data (64 files in `ecog/ECoG_ch{i}.mat` where i ranges from 1 to 64)

## Downloading the Data

The datasets are hosted on Zenodo. To download them:

1. Make sure you have the required Python packages:
```bash
pip install requests tqdm
```

2. Run the download script:
```bash
python download_data.py
```

This will:
- Create the necessary subdirectories (`fmri/`, `meg/`, `ecog/`)
- Download all datasets from Zenodo
- Place them in their respective directories

The script will skip any files that have already been downloaded.

## Directory Structure

After running the download script, you should have the following structure:

```
data/
├── download_data.py
├── README.md
├── fmri/
│   └── Schaefer100_BOLD_HCP.mat
├── meg/
│   └── MEGdataset
└── ecog/
    ├── ECoG_ch1.mat
    ├── ECoG_ch2.mat
    ...
    └── ECoG_ch64.mat
```

## Dataset Details

- **FMRI Dataset**: Contains BOLD signals from the Human Connectome Project using the Schaefer 100-region parcellation
- **MEG Dataset**: Contains magnetoencephalography recordings
- **ECoG Dataset**: Contains 64 channels of electrocorticography recordings