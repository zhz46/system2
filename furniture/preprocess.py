import json
import numpy as np
import pandas as pd
from glob import glob



def data_load(files='../../desktop/data/*.json'):
    # read all furniture jsons
    data = []
    for file_name in glob('../../desktop/data/*.json'):
        with open(file_name) as f:
            temp = json.load(f)
            for sku in temp:
                # read parent product list and product list
                parent_prod = list(sku['elasticsearch_result']['parentProducts'].keys())
                prod = list(sku['elasticsearch_result']['products'].keys())
                image = sku['main_image']['raw']
                sku['image_url'] = image
                if parent_prod:
                    sku['parentProducts'] = parent_prod[0]
                    if prod[0] == parent_prod[0]:
                        sku['products'] = prod[1]
                    else:
                        sku['products'] = prod[0]
                else:
                    sku['parentProducts'] = np.nan
                    if prod:
                        sku['products'] = prod[0]
                    else:
                        sku['products'] = np.nan
            data = data + temp
    return data


def pre_process(raw_data):
    # convert list into dataframe
    raw_df = pd.DataFrame(raw_data)
    # combine skus that don't have groups and primary sku(first one) in each group
    no_group_df = raw_df.loc[raw_df.group_id.isnull()]
    group_df = raw_df.loc[raw_df.group_id.notnull()]
    primary_df = group_df.groupby('group_id').apply(lambda x: x.iloc[0])
    raw_df = pd.concat([primary_df, no_group_df]).reset_index()

    # select key features
    # features = ['title', 'category_id', 'category_level_0', 'category_level_1',
    #             'brand', 'attributes', 'price_hint', 'description', 'sku_id']
    features = ['products', 'parentProducts', 'brand', 'price_hint', 'title', 'category_id', 'id']
    fts = {}
    for i in range(len(features)):
        fts[features[i]] = i
    df = raw_df[features].copy()

    # pre-process data
    # convert to float
    df.price_hint = df.price_hint.astype(float)
    # fill missing
    df.price_hint.fillna(df.price_hint.median(), inplace=True)
    # drop two skus that do not have 'meaningful' titles
    df = df.drop(df.index[[76774, 113749]])
    return (df, fts)
