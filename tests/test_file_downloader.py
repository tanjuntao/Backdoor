import urllib

import requests
from tqdm import tqdm

url = "http://47.96.163.59:80/download/criteo-train.csv"
local_filename = "criteo-train.csv"


# option 1: use requests
# data = requests.get(url)
# with open(local_filename, 'wb') as f:
#     f.write(data.content)


# option 2: use urllib
chunk_size = 1024
with open(local_filename, "wb") as fh:
    with urllib.request.urlopen(urllib.request.Request(url)) as response:
        with tqdm(total=response.length) as pbar:
            for chunk in iter(lambda: response.read(chunk_size), ""):
                if not chunk:
                    break
                pbar.update(chunk_size)
                fh.write(chunk)
