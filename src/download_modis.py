import os
import earthaccess
import xarray as xr
from pathlib import Path 

SAVE_DIR = "../data/raw"
START_YEAR = 2008
END_YEAR = 2008

# make raw data directory if it doesn't exist already
os.makedirs(SAVE_DIR, exist_ok=True)

def download_data():
    # get environment variables
    user = os.getenv('EARTHDATA_USER')
    password = os.getenv('EARTHDATA_PASS')

    # if environment variables exist, use them for username and password for authentication
    if user and password:
        os.environ ['EARTHDATA_USERNAME'] = user
        os.environ ['EARTHDATA_PASSWORD'] = password
        auth = earthaccess.login(strategy="environment", persist=True)
    else: # otherwise, use .netrc method
        auth = earthaccess.login(persist=True)

    for year in range(START_YEAR, END_YEAR+1):
        file_results = []

        results = earthaccess.search_data(
            short_name = "MODISA_L3m_CHL",
            temporal = (f"{year}-01-01", f"{year}-12-31"),
            granule_name="*8D*4km.nc"
        )

        if (not results):
            print (f"No data found for year {year}.")
            continue

        for result in results:
            if ("8D" in result.__str__()):
                file_results.append(result)

        print (f"Found {len(file_results)} granules. Starting download...")
        filelist = earthaccess.download(file_results, SAVE_DIR)

        for file in filelist:
            try:
                dataset = xr.open_dataset(file, engine="netcdf4")

                dataset = dataset.sel(lat=slice(56.8333, 46.1666), lon=slice(-145.8333, -135.1666))

                save_path = file.with_name(file.name.replace('.nc', '_cropped.nc'))
                dataset.to_netcdf(save_path)
                print(f"Processed: {os.path.basename(save_path)}")

                file = Path(file)
                file.unlink()
                print (f"Deleted: {file.name}")

            except Exception as e:
                print(f"Failed to process {file}: {e}")

if __name__ == "__main__":
    download_data()