import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def save_pickle(fn, o):
    with open(fn, 'wb') as f: pickle.dump(o, f)


def load_pickle(fn):
    with open(fn, 'rb') as f: return pickle.load(f)

# ================================================== #

usa_housing = pd.read_csv("california_housing_train.csv")

X = usa_housing[usa_housing.columns[2:-2]]
y = usa_housing[usa_housing.columns[-1]]
save_pickle("column_names.pkl", usa_housing.columns[2:-2])

# X.latitude = pd.qcut(X.latitude, q=4)
# X.longitude = pd.qcut(X.longitude, q=4)

# save_pickle("latitude.pkl", X.latitude.unique())
# save_pickle("longitude.pkl", X.longitude.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2024)

lm = LinearRegression()
print("Training")
lm.fit(X_train, y_train)

# pickle.dump(lm, open("model.pkl", "wb"))
save_pickle("model.pkl", lm)
print("Saved")