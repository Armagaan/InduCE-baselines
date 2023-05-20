"""Create directories, download datasets and models."""
import urllib.request
from subprocess import call

from gdown import download_folder

url = "https://drive.google.com/drive/folders/1AqJerVjBhch3UwOYnmp00XypeHPzGnSD?usp=share_link"

# Check internet connectivity
def connected(host="http://google.com"):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False
assert connected(), "No internet connection!"

# Download data and the trained models.
download_folder(url, output="random-data")

# Unzip
call("unzip random-data/data.zip && unzip random-data/models && unzip random-data/results.zip && unzip random-data/outputs.zip", shell=True)
call("rm -r random-data", shell=True)
