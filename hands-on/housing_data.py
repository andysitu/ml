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

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
# subtract by mean value and deivde by standard deviation.
# does not restrict values to secpfic range bu less afected by outliers
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()