import gdown
import zipfile
import os

url = 'https://drive.google.com/uc?id=1dOWkgkL0rVAg8PJ8QE9oVkhU2Uy3rtGp'
outputFile = 'resources.zip'

gdown.download(url, outputFile, quiet=False)

with zipfile.ZipFile(outputFile, 'r') as zip_ref:
    zip_ref.extractall()

os.remove(outputFile)