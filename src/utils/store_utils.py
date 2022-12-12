import pickle
import os
import pandas as pd
import json

class StorageHandler(object):

    def __init__(self, dir_store):
        self.dir_store = dir_store

    def store(self, object, name):
        type = name.split(".")[-1]
        if type == 'p':
            pickle.dump(
                object,
                open(
                    os.path.join(self.dir_store, name), "wb"))
        elif type == 'csv':
            object.to_csv(
                os.path.join(self.dir_store, name),
                index=False
            )
        elif type == "json":
            json.dump(
                object, open(
                    os.path.join(self.dir_store, name), "w")
                )
        elif type == "txt":
            with open(os.path.join(self.dir_store, name), "a") as f:
                f.write(json.dumps(object))
                f.write("\n")


    def read(self, name):
        type = name.split(".")[-1]
        if type == 'p':
            object = pickle.load(
                open(
                    os.path.join(self.dir_store, name), "rb"))
        elif type == 'csv':
            object = pd.read_csv(
                os.path.join(self.dir_store, name)
            )
        elif type == 'json':
            object = json.load(
                open(
                    os.path.join(self.dir_store, name)))
        elif type == "txt":
            with open(os.path.join(self.dir_store, name), "r") as f:
                object = f.readlines()
        return object



