"""Create directories, download datasets and models."""
import urllib.request
from subprocess import call

from gdown import download_folder

url = "https://drive.google.com/drive/folders/1lI4Lbol1bg_5tKwHYD_aSeTfU7gN-xxI?usp=share_link"

# Check internet connectivity
def connected(host="http://google.com"):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False
assert connected(), "No internet connection!"

# Download data and the trained models.
download_folder(url, output="cfgnn-data")

# Unzip
call("unzip cfgnn-data/data.zip && unzip cfgnn-data/models && unzip cfgnn-data/results.zip", shell=True)
call("rm -r cfgnn-data", shell=True)
