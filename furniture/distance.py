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
    title_dist = (1 - np.dot(a[len(fts):], b[len(fts):])) * 0.5
    return title_dist


def image_only(a, b, fts):
    image_dist = 1 - np.dot(a[len(fts):], b[len(fts):])
    # image_dist = np.linalg.norm(a[len(fts):] - b[len(fts):])
    return image_dist


def combo_dist(a, b, fts, wt):
    title_dist = (1 - np.dot(a[len(fts): len(fts)+300], b[len(fts): len(fts)+300])) * 0.5
    image_dist = 1 - np.dot(a[(len(fts)+300):], b[(len(fts)+300):])
    return wt * title_dist + (1 - wt) * image_dist


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