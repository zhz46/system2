from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as amis
from sklearn.linear_model import LogisticRegression


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