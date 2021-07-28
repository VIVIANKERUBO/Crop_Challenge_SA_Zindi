# Required libraries
import radiant_mlhub
from radiant_mlhub import Collection
import tarfile
import os
import json
from pathlib import Path
import pandas as pd

os.environ['MLHUB_API_KEY'] = 'b8d7b4412bc94350c6716ab291a3bb64d39b159794347e4244edcceed1dd739b'

collections = [
    'ref_south_africa_crops_competition_v1_train_labels',
    'ref_south_africa_crops_competition_v1_train_source_s1', # Comment this out if you do not wish to download the Sentinel-1 Data
    'ref_south_africa_crops_competition_v1_train_source_s2',
    'ref_south_africa_crops_competition_v1_test_labels',
    'ref_south_africa_crops_competition_v1_test_source_s1', # Comment this out if you do not wish to download the Sentinel-1 Data
    'ref_south_africa_crops_competition_v1_test_source_s2'
]

def download(collection_id):
    print(f'Downloading {collection_id}...')
    collection = Collection.fetch(collection_id)
    path = collection.download('.')
    tar = tarfile.open(path, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(path)

def resolve_path(base, path):
    return Path(os.path.join(base, path)).resolve()
def load_df(collection_id):
    collection = json.load(open(f'{collection_id}/collection.json', 'r'))
    rows = []
    item_links = []
    for link in collection['links']:
        if link['rel'] != 'item':
            continue
        item_links.append(link['href'])
        
    for item_link in item_links:
        item_path = f'{collection_id}/{item_link}'
        current_path = os.path.dirname(item_path)
        item = json.load(open(item_path, 'r'))
        tile_id = item['id'].split('_')[-1]
        for asset_key, asset in item['assets'].items():
            rows.append([
                tile_id,
                None,
                None,
                asset_key,
                str(resolve_path(current_path, asset['href']))
            ])
            
        for link in item['links']:
            if link['rel'] != 'source':
                continue
            link_path = resolve_path(current_path, link['href'])
            source_path = os.path.dirname(link_path)
            try:
                source_item = json.load(open(link_path, 'r'))
            except FileNotFoundError:
                continue
            datetime = source_item['properties']['datetime']
            satellite_platform = source_item['collection'].split('_')[-1]
            for asset_key, asset in source_item['assets'].items():
                rows.append([
                    tile_id,
                    datetime,
                    satellite_platform,
                    asset_key,
                    str(resolve_path(source_path, asset['href']))
                ])
    return pd.DataFrame(rows, columns=['tile_id', 'datetime', 'satellite_platform', 'asset', 'file_path'])

import utils
from utils import load_df

#for c in collections:
    #download(c)

train_df = load_df('ref_south_africa_crops_competition_v1_train_labels')
test_df = load_df('ref_south_africa_crops_competition_v1_test_labels')


