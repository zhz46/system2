from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as amis


def cluster_eval(df, k_clust):
    # encode products
    le = LabelEncoder()
    products_group = le.fit_transform(df.products)
    df = df.drop('products', axis=1)
    train = df.values

    # kmeans
    kmeans = KMeans(n_clusters=k_clust)
    cluster_group = kmeans.fit_predict(train)

    return (ari(products_group, cluster_group), amis(products_group, cluster_group))


# load data
# lsi_df = pd.read_csv("lsi_mat.csv", sep=",")
# centroid_df = pd.read_csv("centroid_mat.csv", sep=",")
# wa_df = pd.read_csv("wa_mat.csv", sep=",")

# start k-fold validation to tune hyperparameters
def tune_parameter(X, y, clf, parameters):
    gs = GridSearchCV(clf, param_grid=parameters, scoring='roc_auc', cv=10)
    gs.fit(X, y)
    return gs.best_params_, gs.best_score_, gs.cv_results_


def train_predict(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    return(clf.score(X_test, y_test),
           matthews_corrcoef(y_test, y_pred),
           roc_auc_score(y_test, y_pred_prob))


# Logistic Regression with ridge regularization
# cross validation to learn optimal tuning parameter
# from sklearn.linear_model import LogisticRegression
# clf_lr = LogisticRegression(penalty='l2')
# pars_lr = {'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]}
# tune_best, auc_best, cv_result = tune_parameter(X_train, y_train, clf_lr, pars_lr)
# print('optimal tuning parameter :{}\n'
#       'largest cross validation AUC: {}'.format(tune_best, auc_best))
#
#
# # train and predict based on learned tuning parameter
# clf_lr = LogisticRegression(penalty='l2', **tune_best)
# acc, mcc, auc = train_predict(X_train, y_train, X_test, y_test, clf_lr)
# print('Accuracy rate : {}\n'
#       'Matthews correlation coefficient : {}\n'
#       'Area under the curve : {}'.format(acc, mcc, auc))


target = 'products'

# encode categorical variables
le = LabelEncoder()
products_group = le.fit_transform(df.products)
products_group = label_binarize(products_group, classes=range(70)) #optional


# Split data to training and testing set
y_all = products_group
X_all = df.drop(target, 1).values
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.3, random_state=11)


# Logistic Regression with ridge regularization
# cross validation to learn optimal tuning parameter
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# train and predict based on learned tuning parameter
clf = LogisticRegression(penalty='l2', C=0.4, solver='sag')
clf = OneVsRestClassifier(clf)

clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)

clf.score(X_test, y_test)
roc_auc_score(y_test, y_prob)
log_loss(y_test, y_prob)


