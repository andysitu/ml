from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    datasets_dir = Path("datasets")  

    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=datasets_dir)

    csv_path = datasets_dir / "housing" / "housing.csv"  
    return pd.read_csv(Path(csv_path))

housing = load_housing_data()

# housing.info()

# housing.hist(bins=50, figsize=(12, 8))

# plt.show()

# Splitting data randomly. Not good way to build testing data set
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutations(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set, test_set = train_test_split(housing, test_size=0.2, random_state= 42)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0,grid=True)
# plt.xlabel("Incom category")
# plt.ylabel("Number of districts")
# plt.show()

# from sklearn.model_selection import StratifiedShuffleSplit

# splitter = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=42)
# strat_slits=[]
# for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#     strat_train_set_n = housing.iloc[train_index]
#     strat_test_set_n = housing.iloc[test_index]
#     strat_slits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"] ,random_state=42
)

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in(strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1,inplace=True)

housing = strat_train_set.copy()

# housing.plot(kind="scatter",x="longitude",y="latitude",grid=True,alpha=0.2)
# plt.show()

# housing.plot(kind="scatter",x="longitude",y="latitude",grid=True,
#              s=housing["population"]/100,label="population",
#              c="median_house_value",cmap="jet",colorbar=True,
#              legend=True,sharex=False,figsize=(10,7))
# plt.show()

# Find standard correlation coefficient
# corr_matrix = housing.corr(numeric_only=True)
# correlation_values = corr_matrix["median_house_value"].sort_values(ascending=False)
# print(correlation_values)

# scatter plot of attributes against other attributes
# from pandas.plotting import scatter_matrix

# attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=[12,8])
# plt.show()

# housing.plot(kind="scatter",x="median_income",y="median_house_value", alpha=0.1,grid=True)
# plt.show()

housing["rooms_per_house"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["people_per_house"] = housing["population"]/housing["households"]

# # Correlation with the new columns added in
# corr_matrix = housing.corr(numeric_only=True)
# values = corr_matrix["median_house_value"].sort_values(ascending=False)
# print(values)

# revert to clean training set

# drop creates copy but doesn't affect existing value
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# clean the data

# # clean corresponding districts
# housing.dropna(subset=["total_bedrooms"], inplace=True)

# # get rid of the whole attribute
# housing.dropna(subset=["total_bedrooms"], inplace=True)

# # set missing values to median
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace=True)

imputer = SimpleImputer(strategy="median")

# copy only data with numerical attributes since median calculations only 
# work on median attributes
housing_num = housing.select_dtypes(include=[np.number])

# train or fit the imputer on data
imputer.fit(housing_num)

# store statistics that imputer learned from data during fitting
# only to view data. Not necessary for transform
stats = imputer.statistics_
# print(stats)

# print(housing_num.median().values)

# use trained imputer to transform training set by replacing missing values with learned medians
X = imputer.transform(housing_num)

# There is also KNNIMputer and IterativeImputer

#output of transform ins NumPy array so X has neither column names nor index

# Wrap X in DataFrame and recover column names and index from housing_num
housing_tr = pd.DataFrame(X, columns=housing_num.columns, 
                          index=housing_num.index)

# drop outliers
# from sklearn.ensemble import IsolationForest

# isolation_forest = IsolationForest(random_state=42)
# outlier_pred = isolation_forest.fit_predict(X)

# ocean proxmiity contains text, but only a few possible values
housing_cat = housing[["ocean_proximity"]]
# print(housing_cat.head(8))

# transform to numbers so that it's usable
# from sklearn.preprocessing import OrdinalEncoder

# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# print(housing_cat_encoded[:8])
# print(ordinal_encoder.categories_)


from sklearn.preprocessing import OneHotEncoder

# ML algorithms assume that 2 nearby values are more similar than distant values
# (if ordinal Encoder were used, it would 1-5 representing the categories)
# so need to use one-hot encoding to make them all 0 or 1
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# print(housing_cat_1hot)

# This is a space matrix by default. Use toarray() to convert to NumPy array
# housing_cat_1hot.toarray()

# can also set sparse=False

# can use get_dummies from Panda to convert categorical into one-hot representation
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
# print(pd.get_dummies(df_test))

# But OneHotEncoder remembers which cateogires it was trained on
# a = cat_encoder.transform(df_test)

# it still has the other categories
# print(a)

df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
c = pd.get_dummies(df_test_unknown)
# print(c)

# OneNote will detect known cateogry and raise an exception
# on ignore by handle_unknown to "ignore"

cat_encoder.handle_unknown = "ignore"
# cat_encoder.transform(df_test_unknown)

# when fitting with scikit-learn esimtator using DataFrame, the
# # estimator stores column names in feature_names_in attribute
# print(cat_encoder.feature_names_in_)

# print(cat_encoder.get_feature_names_out())

# can using get_feature_names_out to build a DataFrame around transformer's output
# df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown).toarray(),
#                          columns=cat_encoder.get_feature_names_out(),
#                          index=df_test_unknown.index)

# feature scale with min max
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
# Standard scaler
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
# subtract by mean value and deivde by standard deviation.
# does not restrict values to secpfic range bu less afected by outliers
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# radial basis function - mos common is Gaussian RBF
from sklearn.metrics.pairwise import rbf_kernel
age_siml_35 =  rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)


# feature scaling

from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()

scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5] # pretend this is new data

# the prediction is the log of the median house value because it's 
# scaled, so inverse_transform will transform it to the median value

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)


# TransformedTargetRegressor will do it all for you automatically
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)


# Create a log transform - often used for heavy-tailed distribution of data so ml doens't like it
# replacing it with their logarithm value makes it more distributed

from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

# inverse_func argument is optional - for example if you will use TransformedTargetRegressor

# can take hyperparameters as additional arguments (eg using Gaussian RBF)
rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

# no inversion for RBF kernel since there are 2 values at given distance from fixed point

# rbf_kernel does not treat features separately
# if passed array with 2 features, it will measure the 2D distance (Euclidean)
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                   kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_siml = sf_transformer.transform(housing[["latitude", "longitude"]])

# custom transformers can also combine features
ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))


# use custom class for a trainable transformer

# BaseEstimator and avoid *args and **kwargs - gets get_params and set_params()
# TransformerMixin gets fit_transform which calls fit() and then transform()

from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils.validation import check_array, check_is_fitted

# class StandardScalerClone(BaseEstimator, TransformerMixin):
#     # no *args or **kwargs
#     def __init__(self, with_mean=True):
#         self.with_mean = with_mean

#     def fit(self, X, y=None):  # y is required even if not used
#         X = check_array(X)  # checks that X is an array with finite float values
#         self.mean_ = X.mean(axis=0)
#         self.scale_ = X.std(axis=0)
#         self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
#         return self  # always return self!

#     def transform(self, X):
#         check_is_fitted(self)  # looks for learned attributes (with trailing _)
#         X = check_array(X)
#         assert self.n_features_in_ == X.shape[1]
#         if self.with_mean:
#             X = X - self.mean_
#         return X / self.scale_
    

# custom transformer can use other estimators

# customer transformer that uses KMeans clusterer in fit()
# to idenify main clusters in training data and rbf_kernel in
# transform() to measure how similar each sample is to cluster center

from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=1, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
    
    def transform(self, X):
        # gaussian centered fitting around cluster centers and find out similiarity
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# # compares with the 10 clusters with Gaussian RBF to compare geographic similarity
# cluster_simil = ClusterSimilarity(n_clusters=10,gamma=1,random_state=30)
# similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
#                 sample_weight=housing_labels)

# print(similarities[:3].round(2))


# # transformation pipelines

from sklearn.pipeline import Pipeline

# # names can be anything but must be unique. esimators must all be transformers
# # except last one which can be transformer, predictor, or estimator
# num_pipeline = Pipeline([
#     ("impute", SimpleImputer(strategy="median")),
#     ("standardize", StandardScaler())
# ])

# # use make_pipeline will name it for you
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# housing_num_prepared = num_pipeline.fit_transform(housing_num)
# housing_num_prepared[:2].round(2)

# df_housing_num_prepared = pd.DataFrame(
#     housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
#     index=housing_num.index
# )

# singler transfer capable of handling all columns by applying appropriate transformations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# define list of numerical and categorical column names
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

#construct ColumnTransformer
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

# requires list of triplets: name, transformer, and list of names or indices of columns
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# Use make_column_selector, which does the same
# use make_column_transformer if you don't care about the name

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)

preprocessing.get_feature_names_out()

# train basic linear regression model

from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

# trying on training set

# predicted values
housing_predictions = lin_reg.predict(housing)
# actual values
housing_predictions[:5].round(-2)  # -2 = rounded to the nearest hundred

# compute error ratio
error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))


# Need to use root_mean_squared_error(labels, predictions)
# because RMSE was used
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(labels, predictions):
        return mean_squared_error(labels, predictions, squared=False)

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
lin_rmse
# 68687.89176589991 (pretty high value)

# try using DecisionTreeRegressor for a more accurate model

from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print(tree_rmse)
# 0.0 - model overfitting the data

#k fold cross validation - split into 10 nonoverlapping subsets called folds

from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
# scoring function so higher value is beter
print(pd.Series(tree_rmses).describe())

# it cross-validates and can provide STD but this requires training several times
# High RMSE but low training error (0) so overfitting

# # RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# forest_reg = make_pipeline(preprocessing,
#                            RandomForestRegressor(random_state=42))
# forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
#                                 scoring="neg_root_mean_squared_error", cv=10)

# print(pd.Series(forest_rmses).describe())
# # 47038.092799# 17474 RMSE so there is room for improvement

# try different models

# # grid search
# # uses cross-validation to evaluation all possible combinations of hyperparameters
# from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
# param_grid = [
#     {'preprocessing__geo__n_clusters': [5, 8, 10],
#      'random_forest__max_features': [4, 6, 8]},
#     {'preprocessing__geo__n_clusters': [10, 15],
#      'random_forest__max_features': [6, 8, 10]},
# ]
# grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
#                            scoring='neg_root_mean_squared_error')
# grid_search.fit(housing, housing_labels)

# # get output, keys - hyperparameters available for tuning
# print(str(full_pipeline.get_params().keys())[:1000] + "...")

# cv_res = pd.DataFrame(grid_search.cv_results_)
# cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# # make the DataFrame look nicer
# cv_res = cv_res[["param_preprocessing__geo__n_clusters",
#                  "param_random_forest__max_features", "split0_test_score",
#                  "split1_test_score", "split2_test_score", "mean_test_score"]]
# score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
# cv_res.columns = ["n_clusters", "max_features"] + score_cols
# cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

# cv_res.head()

# randomized search
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

print(rnd_search.fit(housing, housing_labels))

cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
print(cv_res.head())

# analyze best models
final_model = rnd_search.best_estimator_  # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)

sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),
           reverse=True)


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)


from scipy import stats

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse,
                              confidence_level=confidence, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval

rmse_lower, rmse_upper

# joblib
import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")

# example script that would run in production
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

#class ClusterSimilarity(BaseEstimator, TransformerMixin):
#    [...]

final_model_reloaded = joblib.load("my_california_housing_model.pkl")

new_data = housing.iloc[:5]  # pretend these are new districts
predictions = final_model_reloaded.predict(new_data)
