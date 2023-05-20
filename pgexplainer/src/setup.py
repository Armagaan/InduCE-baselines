"""Create directories, download datasets and models."""
import urllib.request
from subprocess import call

from gdown import download_folder

url = "https://drive.google.com/drive/folders/1TuTpPbEGMXkE557Vgve7b-yRje2p4e1t?usp=share_link"

# Check internet connectivity
def connected(host="http://google.com"):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False
assert connected(), "No internet connection!"

# Download data and the trained models.
download_folder(url, output="pgexplainer-data")

# Unzip
call("unzip pgexplainer-data/data.zip && unzip pgexplainer-data/models", shell=True)
call("rm -r pgexplainer-data", shell=True)
