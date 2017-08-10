import pandas as pd
import numpy as np


def prod_process(a, b, fts):
    # calculate products and parentProducts distance
    # either one is missing
    if pd.isnull(a[fts['products']]) or pd.isnull(b[fts['products']]):
        prod_dist = 0.8
    # same products
        prod_dist = 0
    # one's product is same as another's parentProduct
    elif a[fts['products']] == b[fts['parentProducts']] or b[fts['products']] == a[fts['parentProducts']]:
        prod_dist = 0.1
    # one's parentProduct is part of another's product
    elif (pd.notnull(a[fts['parentProducts']]) and a[fts['parentProducts']] in b[fts['products']]) or \
            (pd.notnull(b[fts['parentProducts']]) and b[fts['parentProducts']] in a[fts['products']]):
        prod_dist = 0.3
    # one's product is part of another's product
    elif a[fts['products']] in b[fts['products']] or b[fts['products']] in a[fts['products']]:
        prod_dist = 0.3
    else:
        prod_dist = 1
    return prod_dist


# price processing
def price_process(a, b, fts):
    i = a[fts['price_hint']]
    j = b[fts['price_hint']]
    if j < i:
        result = (i - j)/(i + j)
    elif j > 1.1 * i:
        result = (j - 1.1 * i)/(i + j)
    else:
        result = 0
    return result


# brand processing
def brand_process(a, b, fts):
    if a[fts['brand']] == b[fts['brand']]:
        return 0
    return 1


# calculate title distance only
def title_only(a, b, fts):
    title_dist = (1 - np.dot(a[fts['title'][0]: fts['title'][1]], b[fts['title'][0]: fts['title'][1]])) * 0.5
    return title_dist


def image_only(a, b, fts):
    image_dist = 1 - np.dot(a[fts['image']:], b[fts['image']:])
    # image_dist = np.linalg.norm(a[len(fts):] - b[len(fts):])
    return image_dist


def combo_dist(a, b, fts, wt):
    title_dist = title_only(a, b, fts)
    title_wt = wt['title_wt']
    if wt['prod_wt'] != 0:
        prod_dist = prod_process(a, b, fts)
        prod_wt = wt['prod_wt']
    else:
        prod_dist = 0
        prod_wt = 0
    if wt['image_wt'] != 0:
        image_dist = image_only(a, b, fts)
        image_wt = wt['image_wt']
    else:
        image_dist = 0
        image_wt = 0
    if wt['brand_wt'] != 0:
        brand_dist = brand_process(a, b, fts)
        brand_wt = wt['brand_wt']
    else:
        brand_dist = 0
        brand_wt = 0
    if wt['price_wt'] != 0:
        price_dist = price_process(a, b, fts)
        price_wt = wt['price_wt']
    else:
        price_dist = 0
        price_wt = 0
    wt_list = [title_wt, image_wt, prod_wt, brand_wt, price_wt]
    dist_list = [title_dist, image_dist, prod_dist, brand_dist, price_dist]
    distance = np.dot(wt_list, dist_list)
    return distance


# calculate weighted distance
def mixed_dist(a, b, fts, prod_wt=0.5, brand_wt=0.2, title_wt=0.2, price_wt=0.1):
    # calculate title_dist
    title_dist = title_only(a, b, fts)
    # calculate prod_dist
    prod_dist = prod_process(a, b, fts)
    # calculate price_dist
    price_dist = price_process(a, b, fts)
    # calculate brand_dist
    brand_dist = brand_process(a, b, fts)
    distance = np.dot([prod_wt, brand_wt, title_wt, price_wt],
                  [prod_dist, brand_dist, title_dist, price_dist])
    return distance