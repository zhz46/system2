import numpy as np
import pandas as pd
import sqlite3
from multiprocessing import Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as amis
from sklearn.linear_model import LogisticRegression

from distance import combo_dist, mixed_dist



def cluster_eval(X_all, y_all, k_clust):
    # encode products
    le = LabelEncoder()
    products_group = le.fit_transform(y_all)
    # kmeans
    kmeans = KMeans(n_clusters=k_clust)
    cluster_group = kmeans.fit_predict(X_all)
    return (ari(products_group, cluster_group), amis(products_group, cluster_group))


# start k-fold validation to tune hyperparameters
# def tune_parameter(X, y, clf, parameters):
#     gs = GridSearchCV(clf, param_grid=parameters, scoring='log_loss', cv=10)
#     gs.fit(X, y)
#     return gs.best_params_, gs.best_score_, gs.cv_results_


def train_predict(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    return(clf.score(X_test, y_test),
           log_loss(y_test, y_prob))


def class_eval(X_all, y_all, **pars):
    # encode categorical variables
    le = LabelEncoder()
    y_all = le.fit_transform(y_all)
    # Split data to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=11)

    # Logistic Regression with ridge regularization
    # cross validation to learn optimal tuning parameter
    # clf = LogisticRegression(penalty='l2', solver='sag', multi_class='multinomial')
    # tune_best, log_best, cv_result = tune_parameter(X_train, y_train, clf, pars)
    # print('optimal tuning parameter :{}\n'
    #     'best log loss: {}'.format(tune_best, log_best))

    # train and predict based on learned tuning parameter
    clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', **pars)
    # clf = KNeighborsClassifier()
    acc, ls = train_predict(X_train, y_train, X_test, y_test, clf)
    return (acc, ls)


def weight_series(length, ini=0.95):
    weights = []
    for i in range(length):
        weights.append(ini**i)
    return np.array(weights)


def co_view_select(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    # # store table column names
    # c.execute(
    #     "SELECT * FROM dIIRtCk LIMIT 5"
    #     )
    # co_view = c.fetchall()
    # names = [description[0] for description in c.description]
    # query furniture co-view data
    c.execute(
        "select skuY, skuX, ggXY "
        "from dIIRtCk "
        "where nodeY like '18000%' and nodeX like '18000%' "
        "and nodeX = nodeY "
        "order by skuX "
        )
    co_view = c.fetchall()
    return co_view


def generate_map(df, co_view):
    # secondonary map to primary
    id_list = df.sku_id.values
    sec2pri = {}
    group_map = df.groupby('group_id').groups
    for values in group_map.values():
        for i in range(1, len(values)):
            sec2pri[id_list[values[i]]] = id_list[values[0]]

    # build a co-view map for each sku => top co-view score skus
    co_view_map = {}
    sku_ids = set(id_list)
    for pair in co_view:
        if pair[2] is None:  # no co-view score
            continue
        if pair[1] not in sku_ids or pair[0] not in sku_ids:  # sku_id not exist
            continue
        idx = pair[1]
        if idx not in co_view_map:
            co_view_map[idx] = {}
        if pair[0] in sec2pri:
            idy = sec2pri[pair[0]]
        else:
            idy = pair[0]
        if idy in co_view_map[idx]:
            if pair[2] > co_view_map[idx][idy]:
                co_view_map[idx][idy] = pair[2]
            else:
                continue
        else:
            co_view_map[idx][idy] = pair[2]

    # replace co_view score by ranked score
    for idx in co_view_map:
        sort_arr = sorted(co_view_map[idx], key=lambda x : x[1], reverse=True)
        weight_arr = weight_series(len(sort_arr))
        for i in range(len(sort_arr)):
            co_view_map[idx][sort_arr[i]] = weight_arr[i]

    return co_view_map, sec2pri


def df_sort(df):
    # divide df to 3 parts
    no_group_df = df.loc[df.group_id.isnull()]
    group_df = df.loc[df.group_id.notnull()]
    primary_df = group_df.groupby('group_id').apply(lambda x: x.iloc[0])
    second_df = group_df.groupby('group_id').apply(lambda x: x.iloc[1:])

    # generate candidate df for calculation
    candidate_df = pd.concat([primary_df, no_group_df]).reset_index(drop=True)
    candidate_df = candidate_df.sort_values(by='category_id').reset_index(drop=True)

    # build map
    category_map = candidate_df.groupby('category_id').groups
    for key, value in category_map.items():
        value = value[0], value[-1] + 1
        category_map[key] = value

    # stack everything for title process
    new_df = pd.concat([candidate_df, second_df]).reset_index(drop=True)
    return new_df, category_map


# calculate importance score for each sku
def score(idx, candidate_mat, second_mat, fts, sec2pri, category_map, co_view_map, wt):
    # get co_view dict for the query sku
    co_view_dict = co_view_map[idx]
    # get query data
    if idx in sec2pri:
        cal_data = second_mat[second_mat[:,fts['sku_id']] == idx]
    else:
        cal_data = candidate_mat[candidate_mat[:, fts['sku_id']]== idx]
    cal_data = cal_data.reshape(cal_data.shape[1],)
    cate_id = cal_data[fts['category_id']]
    cate_index = category_map[cate_id]
    # get candidate data, with same category_id and not itself
    candidate_data = candidate_mat[cate_index[0]:cate_index[1]]

    # calculate similarity for each candidate
    dist_mat = np.apply_along_axis(combo_dist, axis=1, arr=candidate_data, b=cal_data, fts=fts, wt=wt)
    # dist_mat = np.apply_along_axis(mixed_dist, axis=1, arr=candidate_data, b=cal_data, fts=fts)

    # sort candidate data
    index_arr = np.argsort(dist_mat)
    candidate_sort = candidate_data[index_arr]
    scores = weight_series(len(candidate_sort))

    # find idy's score
    weight_sum = 0
    full_sum = 0
    for idy, weight_y in list(co_view_dict.items()):
        score_y = scores[candidate_sort[:, fts['sku_id']] == idy]
        if not score_y:
            continue
        weight_sum += weight_y * score_y[0]
        full_sum += weight_y ** 2
    return weight_sum, full_sum


# parallel computing
def parallel(func, chunk, p=6):
    pool = Pool(processes=p)
    result = pool.map(func, chunk)
    pool.close()
    return result
