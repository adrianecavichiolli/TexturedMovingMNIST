""" 
    download_brodatz_textures.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Download script to download all Brodatz textures.

    Code from @MaxOSmith 
"""
import os

URL = "http://www.ux.uis.no/~tranden/brodatz/D{id}.gif"
OUT_PATH = "textures"

def main():
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    out_fname_pattern = os.path.join(OUT_PATH, 'D{id}.gif')

    for i in range(120):
        os.system("wget {} -O {}".format(
            URL.format(id=i), out_fname_pattern.format(id=i)))


if __name__ == "__main__":
    main()
