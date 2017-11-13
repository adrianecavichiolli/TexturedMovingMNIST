""" 
    download_brodatz_textures.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Download script to download all Brodatz textures.

    Code from @MaxOSmith 
"""
import os

URL = "http://www.ux.uis.no/~tranden/brodatz/D{id}.gif"
OUT_PATH = "textures/brodatz/D{id}.gif"

def main():
    for i in range(120):
        os.system("wget {} -O {}".format(
            URL.format(id=i), OUT_PATH.format(id=i)))


if __name__ == "__main__":
    main()
