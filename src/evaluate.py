import os
import sys
sys.path.append(os.path.join(".."))
from src.utils.store_utils import StorageHandler
store_handler = StorageHandler(dir_store=os.path.join("..", 'data'))
import scipy
from lightfm import LightFM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from src.utils.metrics import mapk
from src.utils.metadata_embeddings import MetaDataEmbeddings
import warnings
import numpy as np
warnings.filterwarnings('ignore')

NUM_THREADS = 2
NUM_COMPONENTS = 40
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6
LEARNING_RATE = 0.05


dnn_config = {
    "num_epochs": NUM_EPOCHS,
    "verbose": 2,
    "mini_batch_size": 9000,
    "learning_rate": LEARNING_RATE,
    "item_layers": [
        {"name": "embedding", "output_dim": NUM_COMPONENTS, "reduction_sum": 1, "l2": ITEM_ALPHA}
    ],
    "user_layers": [
        {"name": "embedding", "output_dim": NUM_COMPONENTS, "reduction_sum": 1, "l2": ITEM_ALPHA}
    ],
    "activation": "sigmoid",
    "beta1": 0.5
}


def evaluate_LightFM():
    train = store_handler.read("train.csv").fillna(0).values
    item_features = store_handler.read('item_features_dummies.csv').values
    user_features = store_handler.read('user_features_dummies.csv').values

    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    user_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS,
                    learning_rate=LEARNING_RATE)
    model = model.fit(scipy.sparse.csr_matrix(train),
                      item_features=scipy.sparse.csr_matrix(item_features),
                      user_features=scipy.sparse.csr_matrix(user_features),
                      epochs=NUM_EPOCHS,
                      num_threads=NUM_THREADS)

    test = store_handler.read("test.csv").fillna(0).values
    all_preds = []
    all_actual = []
    for i in range(len(test)):
        user_test_items = list(np.where(test[i] == 1)[0])
        if len(user_test_items) == 0: continue
        preds_scores = model.predict(
            user_ids=i,
            item_ids=np.arange(train.shape[1]),
            item_features=scipy.sparse.csr_matrix(item_features),
            user_features=scipy.sparse.csr_matrix(user_features)
        )
        preds_scores[np.where(train[i] == 1)[0]] = -1000
        all_preds.append(sorted(range(len(preds_scores)), key=lambda k: preds_scores[k], reverse=True))
        all_actual.append(user_test_items)
    print("LightFM first 10 users recos:")
    for i in range(10):
        print(all_preds[i])
    print("LightFM MAP@3: {:.3f}".format(mapk(all_actual, all_preds, k=3)))


def evaluate_MetaDataEmbeddings():
    users = store_handler.read("users.p")
    items = store_handler.read("items.p")
    y = store_handler.read("y.p")

    dnn_config['user_layers'][0].update({'input_dim': np.max(users) + 1})
    dnn_config['item_layers'][0].update({'input_dim': np.max(items) + 1})
    dnn_config['user_layers'][0].update({'input_length': users.shape[1]})
    dnn_config['item_layers'][0].update({'input_length': items.shape[1]})

    meta_data_embds = MetaDataEmbeddings(config=dnn_config)

    meta_data_embds.train(
        users, items, y,
        user_features=store_handler.read('user_features_for_ebds.p').values,
        item_features=store_handler.read('item_features_for_ebds.p').values
    )

    item_biases, item_factors = meta_data_embds.get_item_representations()
    user_biases, user_factors = meta_data_embds.get_user_representations()

    train = store_handler.read("train.csv").fillna(0).values
    test = store_handler.read("test.csv").fillna(0).values
    all_preds = []
    all_actual = []
    for i in range(len(test)):
        user_test_items = list(np.where(test[i] == 1)[0])
        if len(user_test_items) == 0: continue
        preds_scores = user_factors[i].dot(item_factors.T)
        preds_scores += user_biases[i]
        preds_scores += item_biases.reshape(-1, )
        preds_scores[np.where(train[i] == 1)[0]] = -1000
        all_preds.append(sorted(range(len(preds_scores)), key=lambda k: preds_scores[k], reverse=True))
        all_actual.append(user_test_items)
    print("MetaDataEmbeddings first 10 users recos:")
    for i in range(10):
        print(all_preds[i])
    print("MetaDataEmbeddings MAP@3: {:.3f}".format(mapk(all_actual, all_preds, k=3)))

    item_space = (item_biases + item_factors)
    sample_of_users_with_2_transactions = np.where(np.sum(train, axis=1) == 2)[0][:3]
    user_space = user_biases[sample_of_users_with_2_transactions] + user_factors[sample_of_users_with_2_transactions]

    X_embedded = TSNE(n_components=2, n_iter=2000, learning_rate=300, n_iter_without_progress=400).fit_transform(
        np.append(item_space, user_space, axis=0))

    xx = [X_embedded[i][0] for i in range(len(X_embedded))]
    yy = [X_embedded[i][1] for i in range(len(X_embedded))]

    plt.figure(figsize=(10, 5))

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    colors = {i: sorted_names[i * 12] for i in range(len(xx))}
    for i in range(len(item_space)):
        plt.scatter(xx[i], yy[i], alpha=0.5, label="item_{}".format(i), c=colors[i])
    c = 0
    for i in range(len(item_space), len(xx)):
        plt.scatter(xx[i], yy[i], alpha=1, label="user_{}".format(sample_of_users_with_2_transactions[c]), c=colors[i], s=200)
        c += 1
    plt.legend(bbox_to_anchor=(1.15, 1.15))
    plt.title('Users-Items Space')
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "data", "Users&ItemsSpace.png"))
    for i in sample_of_users_with_2_transactions:
        preds_scores = user_factors[i].dot(item_factors.T)
        preds_scores += user_biases[i]
        preds_scores += item_biases.reshape(-1, )
        preds_scores[np.where(train[i] == 1)[0]] = -1000
        preds = sorted(range(len(preds_scores)), key=lambda k: preds_scores[k], reverse=True)
        print("User {}, had bought items {}. Top 2 recos are: {}".format(i, np.where(train[i] == 1)[0], preds[:2]))

if __name__ == '__main__':
    # evaluate_LightFM()
    evaluate_MetaDataEmbeddings()
