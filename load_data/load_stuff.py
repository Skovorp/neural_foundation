import concurrent.futures
import requests
import pandas as pd
from tqdm import tqdm 

folder = 'data/'



df = pd.read_excel(io='/home/ubuntu/NEIRY_ANONYMOUS_DATA_FROM_01_07_2023_TO_20_02_2024.xlsx')
urls = df['preSignedUrl'].tolist()
urls = urls[45990:]

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


with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    list(tqdm(executor.map(download_file, urls), total=len(urls)))