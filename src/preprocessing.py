import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import random
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),  ".."))
from src.utils.store_utils import StorageHandler
import random
store_handler = StorageHandler(dir_store=os.path.join( '..', 'data'))
warnings.filterwarnings('ignore')

use_item_features = True
use_user_features = True
use_negative_votes = False
fill_non_transactions_with_zero = False
pd.set_option('display.max_columns', None)


def read_data():

    offers = store_handler.read("offers.csv")
    if use_negative_votes:
        offers['rate'] = offers.apply(lambda x: 1 if x['event_type'] == 'offer completed' else -1, axis=1)
    else:
        offers['rate'] = offers.apply(lambda x: 1 if x['event_type'] == 'offer completed' else 0, axis=1)
    transactions = offers.groupby(['person', 'offer_id'], as_index=False)['rate'].max()
    transactions = transactions.pivot(index='person', columns='offer_id', values='rate').reset_index()

    profile = store_handler.read("profile.csv")
    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')
    profile['lifetime'] = (profile['became_member_on'].max() - profile['became_member_on']).dt.days
    profile.loc[profile['income'].isnull(), 'income'] = profile['income'].median()
    profile = profile[profile['id'].isin(transactions['person'].unique())].sort_values(by="id").reset_index(drop=True)

    print(
        "Percentage of people that had transactions and we do have profile info: {}".format(
            len(set(profile['id'].values).intersection(set(transactions['person'].unique())))/len(transactions['person'].unique()))
    )
    offers_metadata = store_handler.read("offers_metadata.csv")
    offers_metadata = offers_metadata.sort_values(by="id").reset_index(drop=True)


    users = transactions['person'].values
    items = [i for i in transactions if i != 'person']
    transactions = transactions[items].reset_index(drop=True)
    if fill_non_transactions_with_zero:
        transactions = transactions.fillna(0)

    return transactions, profile, offers_metadata


def get_train_test_data(transactions):
    transactions_matrix = transactions.values
    recs = np.where(~np.isnan(transactions_matrix))
    train_index, test_index = train_test_split([(i, j) for i, j in zip(recs[0], recs[1])], test_size=0.3)

    train_matrix = np.empty(transactions.shape)
    train_matrix[:] = np.NaN
    test_matrix = np.empty(transactions.shape)
    test_matrix[:] = np.NaN

    for i, j in train_index:
        train_matrix[i,j] = transactions_matrix[i,j]
    train = pd.DataFrame(train_matrix, columns=transactions.columns)
    if fill_non_transactions_with_zero:
        train = train.fillna(0)
    # Due to the fact that there might be train users without transactions, because they were drawn to the test sample, we randomly asign for each of those users an item with value 0
    selected_rows = train[train.isnull().all(axis=1)].index
    print("Number of train users that left without item: {}".format(len(selected_rows)))
    for i in selected_rows:
        train.loc[i, random.choice(list(train))] = 0



    test_matrix = np.empty(transactions.shape)
    for i, j in test_index:
        test_matrix[i,j] = transactions_matrix[i,j]
    test = pd.DataFrame(test_matrix, columns=transactions.columns)

    print(
        "Amount of records: {}. Amount of buy offer: {}. Amount descline offer: {}".format(
            len(np.where(~np.isnan(transactions_matrix))[0]),
            len(np.where(transactions_matrix == 1)[0]),
            len(np.where(transactions_matrix == -1)[0]))
    )
    return train, test


def transform_features(offers_metadata, profile):
    continues = ['reward', 'difficulty', 'duration']
    binaries = ['mobile', 'social', 'web']
    categorical = ['offer_type']
    for f in continues:
        bins = np.unique(offers_metadata[f].describe()[['min', '25%', '50%', '75%', 'max']].values)
        labels = ['{}_{}'.format(f, i) for i in range(len(bins) - 1)]
        offers_metadata['binned_{}'.format(f)] = pd.cut(offers_metadata[f], bins=bins, labels=labels, include_lowest=True)

    for f in binaries + categorical:
        offers_metadata['binned_{}'.format(f)] = offers_metadata.apply(lambda x: f + "_" + str(x[f]), axis=1)

    continues = ['age', 'lifetime', 'income']
    binaries = []
    categorical = ['gender']

    for f in continues:
        bins = np.unique(profile[f].describe()[['min', '25%', '50%', '75%', 'max']].values)
        labels = ['{}_{}'.format(f, i) for i in range(len(bins) - 1)]
        profile['binned_{}'.format(f)] = pd.cut(profile[f], bins=bins, labels=labels, include_lowest=True)

    for f in binaries + categorical:
        profile['binned_{}'.format(f)] = profile.apply(lambda x: f + "_" + str(x[f]), axis=1)

    profile = profile[['id'] + [i for i in profile if i.startswith("binned_")]]
    offers_metadata = offers_metadata[['id'] + [i for i in offers_metadata if i.startswith("binned_")]]


    profile_mappings = {}

    profile_features = [i for i in profile if i != "id"]


    c = 0
    if use_user_features:
        for i in profile_features:
            profile_mappings[i] = {}
            tmp_unique_values = sorted(profile[i].unique())
            for j in range(len(tmp_unique_values)):
                profile_mappings[i][tmp_unique_values[j]] = c
                c += 1
            profile[i+'_index'] = profile.apply(lambda x: profile_mappings[i][x[i]], axis=1)
    else:
        tmp_unique_values = sorted(profile["id"].unique())
        profile_mappings["id"] = {}
        for j in range(len(tmp_unique_values)):
            profile_mappings["id"][tmp_unique_values[j]] = c
            c += 1
        profile['id_index'] = profile.apply(lambda x: profile_mappings["id"][x['id']], axis=1)

    offers_metadata_mappings = {}
    offers_metadata_features = [i for i in offers_metadata if i != "id"]

    c = 0
    if use_item_features:
        for i in offers_metadata_features:
            offers_metadata_mappings[i] = {}
            tmp_unique_values = sorted(offers_metadata[i].unique())
            for j in range(len(tmp_unique_values)):
                offers_metadata_mappings[i][tmp_unique_values[j]] = c
                c += 1
            offers_metadata[i+'_index'] = offers_metadata.apply(lambda x: offers_metadata_mappings[i][x[i]], axis=1)
    else:
        offers_metadata_mappings["id"] = {}
        tmp_unique_values = sorted(offers_metadata["id"].unique())
        for j in range(len(tmp_unique_values)):
            offers_metadata_mappings["id"][tmp_unique_values[j]] = c
            c += 1
        offers_metadata["id" + '_index'] = offers_metadata.apply(lambda x: offers_metadata_mappings['id'][x['id']], axis=1)

    return offers_metadata, profile


def store_outputs(train, test, profile, offers_metadata):

    if use_item_features:

        features = [i for i in offers_metadata if ((i.startswith('binned')) & (i.endswith('index')))]
        df = offers_metadata[features]
        print("Item features: {}".format(list(df)))
        print(df.values)
        item_features = df.values
        store_handler.store(df, 'item_features_for_ebds.p')

        tmp_dummies = pd.get_dummies(df.astype(str))
        constructed_features = list(tmp_dummies)
        df = pd.concat([df, tmp_dummies], axis=1)
        store_handler.store(df[constructed_features], 'item_features_dummies.csv')
        print("Item Dummy-coded features: {}".format(constructed_features))
        print(df[constructed_features].values)

    if use_user_features:

        features = [i for i in profile if ((i.startswith('binned')) & (i.endswith('index')))]

        df = profile[features]
        user_features = df.values
        print("User features: {}".format(list(df)))
        print(df.values)
        store_handler.store(df, 'user_features_for_ebds.p')

        tmp_dummies = pd.get_dummies(df.astype(str))
        constructed_features = list(tmp_dummies)

        df = pd.concat([df, tmp_dummies], axis=1)
        store_handler.store(df[constructed_features], 'user_features_dummies.csv')
        print("User Dummy-coded features: {}".format(constructed_features))
        print(df[constructed_features].values)



    train_transactions = []

    for i, j in zip(np.where(~train.isnull())[0], np.where(~train.isnull())[1]):

        if use_user_features:
            user = list(user_features[i])
        else:
            user = [i]
        if use_item_features:
            item = list(item_features[j])
        else:
            item = [j]
        train_transactions.append((user, item, int(train.iloc[i, j])))

    train_transactions = random.sample(train_transactions, k=len(train_transactions))

    print("Example of train transaction. UserFeatures-ItemFeatures-Action")
    print(train_transactions[2])

    users = np.array([i[0] for i in train_transactions])
    items = np.array([i[1] for i in train_transactions])
    y = np.array([i[2] for i in train_transactions]).reshape(-1, 1)

    store_handler.store(train_transactions, 'train_transactions.p')
    store_handler.store(train, 'train.csv')
    store_handler.store(test, 'test.csv')
    store_handler.store(offers_metadata, 'item_features.csv')
    store_handler.store(profile, 'user_features.csv')

    store_handler.store(users, 'users.p')
    store_handler.store(items, 'items.p')
    store_handler.store(y, 'y.p')


def preprocess():
    transactions, profile, offers_metadata = read_data()
    train, test = get_train_test_data(transactions)
    offers_metadata, profile = transform_features(offers_metadata, profile)
    store_outputs(train, test, profile, offers_metadata)


if __name__ == "__main__":
    preprocess()
