from scipy.stats import mode
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


def cabin_prefix(X):
    X['Cabin'] = X['Cabin'].fillna('Unknown').str[0]


def fill_age_with_median(X):
    # In case there may be something meaningful about missing age
    X['AgeMissing'] = X['Age'].isnull().astype(int)
    X['Age'] = X['Age'].fillna(X['Age'].median())


def fill_embarked_with_mode(X):
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])


def fill_fare_with_mean(X):
    X['Fare'] = X['Fare'].fillna(X['Fare'].mean())


def remove_id_name_ticket(X, remove_id=True):
    if remove_id:
        del X['PassengerId']
    del X['Name']
    del X['Ticket']


def family_size_vars(X):
    X['TotalFamily'] = X['SibSp'] + X['Parch']
    X['SelfOnly'] = X['TotalFamily'].map(lambda x: 1 if x == 0 else 0)


def categorical_dummies(X, *cols):
    for col in cols:
        dummies = pd.get_dummies(
            X[col], prefix=col, drop_first=True)
        X = pd.concat([X, dummies], axis=1)
        del X[col]
    return X


def feature_importances(clf, X, y):
    clf.fit(X, y)

    features = pd.DataFrame()
    features['feature'] = X.columns
    features['importance'] = clf.feature_importances_

    return features.sort_values(['importance'], ascending=False).reset_index(drop=True)


def plot_feature_importances(importances):
    importances_plt = importances.sort_values(
        ['importance'], ascending=True).reset_index(drop=True)
    ind = importances_plt.index
    plt.barh(ind, importances_plt['importance'], height=.3, align='center')
    plt.ylim(ind.min() + .5, ind.max() + .5)
    plt.yticks(ind, importances_plt['feature'])


class Estimators(object):
    '''
    Estimator object for fitting, storing, and comparing multiple model outputs.
    '''

    def __init__(self):
        lr = LogisticRegression(random_state=1)
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=1)
        gb = GradientBoostingClassifier(learning_rate=.1, n_estimators=1000, random_state=1)
        ab = AdaBoostClassifier(learning_rate=.1, n_estimators=1000, random_state=1)

        self.estimators = [rf, lr, gb, ab]
        self.estimator_names = [
            est.__class__.__name__ for est in self.estimators]

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.scaler = preprocessing.StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)

        for est in self.estimators:
            est.fit(X_train_scaled, self.y_train)

    def select_features(self):
        self.selected_X = []
        self.selected_X_names = []
        # Need to fit estimators first
        
        for est in self.estimators:
            model = SelectFromModel(est, prefit=True)
            new_X = model.transform(self.X_train)
            names = self.X_train.columns.values[model.get_support()]
            self.selected_X.append(new_X)
            self.selected_X_names.append(names)
        print zip(self.estimator_names, self.selected_X_names)
        return zip(self.estimator_names, self.selected_X_names)

    def cv_train_accuracy(self, cv_folds=10):
        self.cv_scores = []
        for est in self.estimators:
            cv_score = np.mean(cross_val_score(
                est, self.X_train, self.y_train, cv=cv_folds, scoring='accuracy'))
            self.cv_scores.append(cv_score)
        print zip(self.estimator_names, self.cv_scores)

    def predict(self, X_test):
        self.X_test = X_test
        X_test_scaled = self.scaler.transform(X_test)

        self.predictions = []
        for est in self.estimators:
            self.predictions.append(est.predict(X_test_scaled))

    def test_accuracy(self, y_test):
        self.y_test = y_test

        self.test_scores = []
        for prediction in self.predictions:
            self.test_scores.append(np.mean(prediction == self.y_test))
        print zip(self.estimator_names, self.test_scores)


def grid_search(est, X, y, params):
    grid = GridSearchCV(est, params,
                        verbose=True,
                        scoring='accuracy',
                        cv=10).fit(X, y)
    print('Best score: {}'.format(grid.best_score_))
    print('Best parameters: {}'.format(grid.best_params_))
    return grid


def get_logit_coef(X, y, cols=None):
    if cols:
        X_fit = X[cols]
    else:
        X_fit = X
    X_fit = sm.add_constant(X_fit)
    logit = Logit(y, X_fit)
    fit = logit.fit()
    print fit.summary()


if __name__ == '__main__':
    # Load train and test data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Separate features and labels
    X = train.copy()
    y = X.pop('Survived')

    # Test data doesn't have labels
    X_test = test.copy()

    cabin_prefix(X)
    cabin_prefix(X_test)
    fill_age_with_median(X)
    fill_age_with_median(X_test)
    fill_embarked_with_mode(X)
    fill_embarked_with_mode(X_test)
    fill_fare_with_mean(X)
    fill_fare_with_mean(X_test)
    remove_id_name_ticket(X)
    remove_id_name_ticket(X_test, remove_id=False)
    family_size_vars(X)
    family_size_vars(X_test)
    X = categorical_dummies(X, 'Pclass', 'Sex', 'Cabin', 'Embarked')
    X_test = categorical_dummies(X_test, 'Pclass', 'Sex', 'Cabin', 'Embarked')

    # clf = RandomForestClassifier(n_estimators=250)
    # importances = feature_importances(clf, X, y)
    # select = list(importances['feature'][:15])
    # plot_feature_importances(importances)

    selector = SelectKBest(chi2, k=15).fit(X, y)
    cols = list(X.columns.values[selector.get_support()])

    # Split training data and holdout data
    X_train, X_val, y_train, y_val = train_test_split(X, y)

    models = Estimators()
    models.train(X_train[cols], y_train)
    # models.cv_train_accuracy()
    models.predict(X_val[cols])
    models.test_accuracy(y_val)

    # models_features = Estimators()
    # models_features.train(X, y)
    # features = models_features.select_features()

    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X[cols])

    output = pd.DataFrame(X_test.pop('PassengerId'))
    X_test_scaled = scaler.transform(X_test[cols])

    gb_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
                 'max_depth': [1, 2, 3],
                 'max_features': [1.0, 0.5, 0.3],
                 'n_estimators': [500]}

    gb_grid = grid_search(GradientBoostingClassifier(), X_scaled, y, gb_params)

    y_pred = gb_grid.predict(X_test_scaled)

    output['Survived'] = y_pred
    output.to_csv('output.csv', index=False)
