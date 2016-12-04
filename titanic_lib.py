from scipy.stats import mode
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


def feature_importances(clf, X, y):
    '''
    Input: classifier (class name), features (df), dependent variable (df)
    Output: feature importances (df)
    '''
    clf.fit(X, y)

    features = pd.DataFrame()
    features['feature'] = X.features
    features['importance'] = clf.feature_importances_

    return features.sort_values(['importance'], ascending=False).reset_index(drop=True)

def plot_feature_importances(importances):
    '''
    Input: importances (df)
    Output: feature importances plot
    '''
    importances_plt = importances.sort_values(
        ['importance'], ascending=True).reset_index(drop=True)
    ind = importances_plt.index
    plt.barh(ind, importances_plt['importance'], height=.3, align='center')
    plt.ylim(ind.min() + .5, ind.max() + .5)
    plt.yticks(ind, importances_plt['feature'])


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


class DataCleaning(object):

    def __init__(self, X, X_test=None):
        '''
        Input: features from training set (df), optional features from test set (df)
        '''
        self.X = X
        self.n_train = X.shape[0]
        self.X_test = X_test

    def combine_data(self):
        '''
        Combine train and test data for processing, if there is test data input
        '''
        self.X = self.X.append(self.X_test)

    def cabin_prefix(self):
        '''
        Trim cabin number and include only letter prefix
        '''
        self.X['Cabin'] = self.X['Cabin'].fillna('Unknown').str[0]

    def fill_age_with_median(self):
        self.X['Age'] = self.X['Age'].fillna(self.X['Age'].median())
        self.X['AgeMissing'] = self.X['Age'].isnull().astype(int)

    def fill_embarked_with_mode(self):
        self.X['Embarked'] = self.X['Embarked'].fillna(self.X['Embarked'].mode()[0])

    def fill_fare_with_mean(self):
        self.X['Fare'] = self.X['Fare'].fillna(self.X['Fare'].mean())

    def man_woman_child(self):
        '''
        Create new grouping of Child, Adult Man, or Adult woman based on Age and Sex
        '''
        def apply_func(age_sex):
            age, sex = age_sex
            if age < 16:
                return 'Child'
            else:
                return 'Woman' if sex == 'female' else 'Man'
        self.X['PersonType'] = self.X[['Age', 'Sex']].apply(apply_func, axis=1)

    def title_from_name(self):
        '''
        Credit to Ahmed BESBES (ahmedbesbes.com) for this clever way of using the the names to extract titles. This maps the various titles into groups: Officer, Royalty, Mrs, Miss, Mr, Master
        '''
        title_dict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

        self.X['Title'] = self.X['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip()).map(title_dict)

    def remove_name_and_ticket(self):
        '''
        Delete Name and Ticket columns as they won't be used in our modeling
        '''
        del self.X['Name']
        del self.X['Ticket']

    def family_size_vars(self):
        '''
        Get total family size by adding siblings/spouse + parents/children
        Identify those who were traveling alone
        '''
        self.X['TotalFamily'] = self.X['SibSp'] + self.X['Parch']
        self.X['SelfOnly'] = self.X['TotalFamily'].map(lambda x: 1 if x == 0 else 0)

    def categorical_dummies(self):
        '''
        Create dummy variables for categorical data so that output data is all numeric and ready for scaling / modeling.
        '''
        cols = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'PersonType', 'Title']
        for col in cols:
            dummies = pd.get_dummies(
                self.X[col], prefix=col) # drop_first=True
            self.X = pd.concat([self.X, dummies], axis=1)
            del self.X[col]
        return self.X

    def clean_data(self, dummies=True):
        '''
        Runs all data cleaning steps
        '''
        if self.X_test is not None:
            self.combine_data()
        self.cabin_prefix()
        self.fill_age_with_median()
        self.fill_embarked_with_mode()
        self.fill_fare_with_mean()
        self.man_woman_child()
        self.title_from_name()
        self.remove_name_and_ticket()
        self.family_size_vars()
        if dummies:
            self.categorical_dummies()

    def output_data(self):
        '''
        Split and output training and test data
        '''
        if self.X_test is not None:
            return self.X[:self.n_train], self.X[self.n_train:]
        else:
            return self.X


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
        self.estimator_names = [est.__class__.__name__ for est in self.estimators]

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.scaler = preprocessing.StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        for est in self.estimators:
            est.fit(self.X_train_scaled, self.y_train)

    def cv_train_accuracy(self, cv_folds=10):
        self.cv_scores = []
        for est in self.estimators:
            cv_score = np.mean(cross_val_score(
                est, self.X_train_scaled, self.y_train, cv=cv_folds, scoring='accuracy'))
            self.cv_scores.append(cv_score)
        output = pd.DataFrame(zip(self.estimator_names, self.cv_scores))
        output.columns = ['Estimator', 'CV Train Accuracy']
        print output

    def predict(self, X_test):
        self.X_test = X_test

        self.X_test_scaled = self.scaler.transform(X_test)

        self.predictions = []
        for est in self.estimators:
            self.predictions.append(est.predict(self.X_test_scaled))


if __name__ == '__main__':
    # Load train and test data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Separate features and labels
    X_train = train.copy()
    y_train = X_train.pop('Survived')

    # Test data doesn't have labels, just make a copy
    X_test = test.copy()

    # Data cleaning / feature engineering
    data = DataCleaning(X_train, X_test)
    data.clean_data(dummies=True)
    X_train, X_test = data.output_data()

    # Train various estimators and compare cross-validated accuracy
    # models = Estimators()
    # models.train(X_train, y_train)
    # models.cv_train_accuracy()

    # gb_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
    #              'max_depth': [1, 2, 4, 6],
    #              'max_features': [1.0, 0.5, 0.3],
    #              'n_estimators': [500]}

    # gb_grid = grid_search(GradientBoostingClassifier(), X_train_scaled, y_train, gb_params)

    # y_pred = gb_grid.predict(X_test_scaled)

    # output = X_test['PassengerId']
    # output['Survived'] = y_pred
    # output.to_csv('output4.csv', index=False)

# EDA
# Thinking
# Feature Engineeering

# Model Selections (could just use cv train; or train test split on the train)

# Hyperparameter Tuning (for a couple models, different parameters)
# - # 2, 4, 6


# # Explaining vs. Predicting
# Feature Selections # maybe not necessary
# - Relaxed Lasso
# Logistic => explaining
# Bad features captured in CV


