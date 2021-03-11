import os
import tempfile
import requests
import py7zr as py7zr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    tmp_archive = tempfile.NamedTemporaryFile(suffix='.7z')
    r = requests.get("https://seafile.zfn.uni-bremen.de/f/7e588dd285de4d7486a6/?dl=1", allow_redirects=True)
    with open(tmp_archive.name, "wb") as f:
        f.write(r.content)
    with py7zr.SevenZipFile(tmp_archive.name, mode='r') as z:
        z.extractall(SCRIPT_DIR)


if __name__ == '__main__':
    main()
