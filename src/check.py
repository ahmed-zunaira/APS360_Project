import requests
from bs4 import BeautifulSoup

url = "https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/2016/0101/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
for link in soup.find_all("a"):
    print(link.get('href'))