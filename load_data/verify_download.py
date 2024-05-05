from tqdm import tqdm
import pandas as pd
import os

df = pd.read_excel(io='/home/ubuntu/NEIRY_ANONYMOUS_DATA_FROM_01_07_2023_TO_20_02_2024.xlsx')
target_list = [x.split('/')[-1] for x in df['preSignedUrl']]
print("len targets:", len(target_list))


found_list = [x.split('--')[0][5:]for x in os.listdir('data')]

print("found len:", len(found_list))
print('diff size1: ', len(set(target_list) - set(found_list)))
print('diff size2: ', len(set(found_list) - set(target_list)))
print("diff:", set(target_list) - set(found_list))


# print(df['preSignedUrl'][df['preSignedUrl'].str.contains('ad45824e-a9cd-4e1c-83c3-036a1851eef4')])

