import os
import glob
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from pathlib import Path

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")

def process_file(file_path):
    dataset = xr.open_dataset(file_path, engine="netcdf4")

    data_np = np.array(dataset['chlor_a'].values)

    dataset.close()

    data_np = np.nan_to_num(data_np)

    mask_path = PROCESSED_DIR / 'land_mask.pt'

    if not mask_path.exists():
        # generating a biinary mask to mask out the coastline
        mask = (data_np > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        torch.save (mask_tensor, mask_path)

    data_np = data_np + 1e-5
    data_np = np.log1p(data_np)

    data_tensor = torch.from_numpy(data_np).float()
    data_tensor = data_tensor.unsqueeze(0)

    save_path = PROCESSED_DIR / file_path.name.replace('_cropped.nc', '.pt').replace('.nc', '.pt')
    torch.save(data_tensor, save_path)
    pass

if __name__ == "__main__":
    os.makedirs (PROCESSED_DIR, exist_ok=True)

    cropped_files = list(RAW_DIR.glob("*_cropped.nc"))

    for file_path in cropped_files:
        try:
            process_file(file_path)
            print (f"Processed: {file_path.name}")
        except Exception as e:
            print (f"Failed to process {file_path.name}: {e}")