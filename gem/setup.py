"""Create directories, download datasets and models."""
import urllib.request
from subprocess import call

from gdown import download_folder

url = "https://drive.google.com/drive/folders/13vZwNpnRLxkEYg_s8Z3BLDiLoDXCtZ-6?usp=share_link"

# Check internet connectivity
def connected(host="http://google.com"):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False
assert connected(), "No internet connection!"

# Download data and the trained models.
download_folder(url, output="gem-data")

# Unzip
call(
    "unzip gem-data/cfgnn_model_weights.zip &&"
    "unzip gem-data/data.zip &&"
    "unzip gem-data/distillation.zip &&"
    "unzip gem-data/explanation.zip &&"
    "unzip gem-data/output.zip",
    shell=True
)
call("rm -r gem-data", shell=True)
