import json
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

__location = None
__data_columns = None
__model = None


def get_price(location,sqft,bath,bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    X = np.zeros(len(__data_columns))
    X[0]=sqft
    X[1]=bath
    X[2]=bhk

    if loc_index>=0:
        X[loc_index] = 1

    return round(__model.predict([X])[0],2)
def get_location_name():
    return __location

def load_artifacts():
    print("Loading saved artifacts")
    global __location
    global __data_columns
    global __model
    with open("./artifacts/columns.json",'r') as f:
        __data_columns = json.load(f)["data_columns"]
        __location = __data_columns[3:]

    with open('./artifacts/banglore_home_price_model.pickle','rb') as f:
        __model = pickle.load(f)
    print("loading...done")


if __name__=='__main__':
    load_artifacts()
    print(get_location_name())
    print(get_price('1st Phase JP Nagar',1000,2,2))
    print(get_price('1st Phase JP Nagar',1000,2,3))
    print(get_price('Indira Nagar',1000,2,2))
    print(get_price('Indira Nagar',1000,3,3))
