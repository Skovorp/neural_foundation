import concurrent.futures
import requests
import pandas as pd
from tqdm import tqdm 

folder = 'data/'

links = [
    # "https://collector.neiry-bci.com/temp-links/252ad207-a4fb-439c-b86d-93865cc5aeb0",
    # "https://collector.neiry-bci.com/temp-links/2fbefa06-0cfd-45ac-b38a-5e61d2330edb",
    # "https://collector.neiry-bci.com/temp-links/33e9eb77-4f11-45b1-b421-109ce054728a",
    # "https://collector.neiry-bci.com/temp-links/d00f9b29-6c8c-41cd-ac33-d90e5a07c752",
    # "https://collector.neiry-bci.com/temp-links/a42f1031-60ba-47e0-a965-26ae3f597db0",
    # "https://collector.neiry-bci.com/temp-links/bbe54d27-7d06-4764-a666-7e91b305b890",
    # "https://collector.neiry-bci.com/temp-links/e3d26289-2734-45fe-b8ce-0012182393b1",
    # "https://collector.neiry-bci.com/temp-links/a10390b3-81e2-4de5-b568-4cd5538b1756",
    # "https://collector.neiry-bci.com/temp-links/6c8ae0e4-42e2-44c7-a9ec-52b87e1c6f83",
    # "https://collector.neiry-bci.com/temp-links/d4ebacfe-a57a-4b23-b341-635706d5c095",
    # "https://collector.neiry-bci.com/temp-links/bcb2720c-32fe-4a7f-a0df-2b6be2abfcdd",
    # "https://collector.neiry-bci.com/temp-links/745afee2-89bb-41a4-b615-e266ce45f905",
    # "https://collector.neiry-bci.com/temp-links/46506edc-d1d9-4e04-8a05-a0b9bfb78233",
    # "https://collector.neiry-bci.com/temp-links/5a06d11d-fc94-467b-b7bc-524f69d75c3a",
    "https://collector.neiry-bci.com/temp-links/ad45824e-a9cd-4e1c-83c3-036a1851eef4"
]


def download_file(url):
    try:
        res = requests.get(url, allow_redirects=False)
        real_link = res.headers['Location']
        file_name = real_link.split('/')[-1].split('?')[0]
        file_name = f"link-{url.split('/')[-1]}--" + file_name
        
        
        response = requests.get(real_link)
        with open(folder + file_name, 'wb') as f:
            f.write(response.content)
    except Exception:
        print(f"Skipping {url}")
        
for l in links:
    download_file(l)
    print('done')