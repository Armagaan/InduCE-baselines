"""Create directories, download datasets and models."""
import urllib.request
from subprocess import call

from gdown import download_folder

url = "https://drive.google.com/drive/folders/1Z1LCq2rq38QXgC11ez7AhxOxzL_FqLxv?usp=share_link"

# Check internet connectivity
def connected(host="http://google.com"):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False
assert connected(), "No internet connection!"

# Download data and the trained models.
download_folder(url, output="cfsqr-data")

# Unzip
call("unzip cfsqr-data/datasets.zip && unzip cfsqr-data/cfgnn_model_weights && unzip cfsqr-data/outputs.zip", shell=True)
call("rm -r cfsqr-data", shell=True)
