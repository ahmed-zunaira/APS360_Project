import os
import requests
import xarray as xr
from bs4 import BeautifulSoup
import datetime

SAVE_DIR = "../data/raw"
START_YEAR = 2016
END_YEAR = 2026

# creating the save directory 
os.makedirs(SAVE_DIR, exist_ok=True)

def get_8days(year):
    """
    Returns a list of strings of calendar days 8 days apart, to match MODIS' formatting
    """
    days = []
    start_date = datetime.datetime(year, 1, 1)

    for i in range(46):
        curr = start_date + datetime.timedelta(days=i*8)

        if (curr.year != year):
            break

        days.append(curr.strftime("%m%d"))
    
    return days

def get_filename (year, day):
    """
    Returns the string MODIS file name for a datset of the provided day of the given year
    """
    url = f"https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/{year}/{day}/"

    response = requests.get(url)

    if response.status_code != 200:
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')

    check_8d = ".8D."
    check_chlor = "chlor_a.4km.nc"
    check_link = ""

    links = soup.find_all("a")
    for link in links:
        curr = link.get('href')
        if curr and check_8d in curr and curr.endswith(check_chlor):
            if "viewers" not in curr:
                check_link = curr
                break
    if (check_link == ""):
        return ""
    
    return check_link

def download_slice(year, day, filename):
    opendap_url = f"https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/{year}/{day}/{filename}"
    local_save_path = os.path.join (SAVE_DIR, filename)

    print ("Processing: {}...".format(filename))

    try:
        # get dataset
        dataset = xr.open_dataset(opendap_url, engine="netcdf4")

        # crop dataset to bc coast
        dataset = dataset.sel(lat=slice(55, 48), lon=slice(-135, -122))

        # save dataset to data folder
        dataset.to_netcdf(local_save_path)

        # close the dataset
        dataset.close()

        pass

    except Exception as e:
        print (f"Failed to process {filename}: {e}")

# main
if __name__ == "__main__":

    for year in range(START_YEAR, END_YEAR + 1):
        days = get_8days(year)

        for day in days:
            # locating the file for the day in specified year
            filename = get_filename(year, day)
            
            # if found, download the file and crop the dataset
            if filename:
                download_slice(year, day, filename)
            else:
                print (f"Could not find a matching file for {year} day {day}.")
