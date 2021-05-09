import pandas as pd
import numpy as np
from scipy import sparse
import os
import random
import pickle as pkl
import torch

from utils import normalize_features, preprocess_user_item_features


def map_data(data):
    """
    将每部电影或者用户的编号从1~N的编号方式转化为0~N-1，并且输出一个字典保存对应关系，还有用户和物品的数量
    Parameters
    ----------
    data : np.int32 arrays
    Returns
    -------
    mapped_data : np.int32 arrays
    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    n = len(uniq)

    return data, id_dict


def split_ratings(ratings):   
    """按照一定比例分割ratings矩阵
    Args:
        ratings ([DataFrame]): [完整并且打乱顺序的rating矩阵]
    Returns:
        [list(dataframe)]: [分割后的rating矩阵]
    """
    split_result = []
    # 分割train-val-test数据集比例
    ratings_num = ratings.shape[0]
    len_train = int(ratings_num*0.85)
    len_val   = int(ratings_num*0.9)

    rating_train = ratings[:len_train]
    rating_val   = ratings[len_train:len_val]
    rating_test  = ratings[len_val:]
    rating_cnt = int(ratings["ratings"].max())
    num_users = int(ratings["u_nodes"].max())
    num_items = int(ratings["v_nodes"].max())
    for i, rating in enumerate([rating_train, rating_val, rating_test]):
        rating_mtx = torch.zeros(rating_cnt, num_users, num_items)
        for index, row in rating.iterrows():
            u = int(row["u_nodes"]-1)
            v = int(row["v_nodes"]-1)
            r = int(row["ratings"]-1)
            rating_mtx[r, u, v] = 1
        split_result.append(rating_mtx)
    return split_result[0], split_result[1], split_result[2], num_users, num_items


def load_data(seed=1234, verbose=True):
    """[summary]

    Args:
        seed (int, optional): [description]. Defaults to 1234.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        num_users : int Number of users and items respectively
        num_items : int
        u_nodes : np.int32 arrays User indices
        v_nodes : np.int32 array item (movie) indices
        ratings : np.float32 array
            User/item ratings s.t. ratings[k] is the rating given by user u_nodes[k] to
            item v_nodes[k]. Note that that the all pairs u_nodes[k]/v_nodes[k] are unique, but
            not necessarily all u_nodes[k] or all v_nodes[k] separately.
        u_features: np.float32 array, or None
            If present in dataset, contains the features of the users.
        v_features: np.float32 array, or None
            If present in dataset, contains the features of the users.
    """    
    
    u_features = None
    v_features = None

    # 获取数据文件路径
    sep = "\:\:"
    files = ['ratings.dat', 'movies.dat', 'users.dat']
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    dtype = {'u_nodes': np.int64, 'v_nodes': np.int64,
             'ratings': np.float32, 'timestamp': np.float64}
    # 读取源数据
    ratings = pd.read_csv(os.path.join(data_dir, files[0]), header=None, sep=sep, \
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], engine="python",\
        converters=dtype)

    movies_df = pd.read_csv(os.path.join(data_dir, files[1]), sep=sep, header=None,
                            names= ['movie_id', 'title', 'genre'], engine='python')

    users_df = pd.read_csv(os.path.join(data_dir, files[2]), sep=sep, header=None,
                            names=['user_id', 'gender', 'age', 'occupation', 'zip-code'], engine='python')
    
    # shuffle ratings 并且分割
    ratings = ratings.sample(frac=1, random_state=seed)
    ratings_train, ratings_val, ratings_test, \
        num_users, num_items = split_ratings(ratings)
    # 转化为nparray
    ratings_array = np.array(ratings)
    # 分别为每一列转化为对应的数据类型
    u_nodes_ratings = ratings_array[:, 0].astype(dtype['u_nodes'])
    v_nodes_ratings = ratings_array[:, 1].astype(dtype['v_nodes'])
    ratings = ratings_array[:, 2].astype(dtype['ratings'])

    u_nodes_ratings, u_dict = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict = map_data(v_nodes_ratings)
    # 重新转换确保类型正确
    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
    ratings = ratings.astype(np.float32)

    
    # 获取所有的电影类别的数量, 每一部电影是多类别的，因此需要分割
    genres = []
    for s in movies_df['genre'].values:
        genres.extend(s.split('|'))
    genres = list(set(genres))
    num_genres = len(genres)  # 共18种类型
    # 为每一个电影类型编号
    genres_dict = {g: idx for idx, g in enumerate(genres)}

    # 构建电影所属类别的特征矩阵
    v_features = np.zeros((num_items, num_genres), dtype=np.float32)
    for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
        # 确保该部电影有评分记录
        if movie_id in v_dict.keys():
            gen = s.split('|')
            for g in gen:
                v_features[v_dict[movie_id], genres_dict[g]] = 1.


    # 提取用户的特征(也就不要user_id列)
    cols = users_df.columns.values[1:]
    cntr = 0
    feat_dicts = []
    for header in cols:
        d = dict()
        # 对特征值去重
        feats = np.unique(users_df[header].values).tolist()
        # 为每个特征值编号(例如性别--{'F': 0, 'M': 1})
        d.update({f: i for i, f in enumerate(feats, start=cntr)})
        feat_dicts.append(d)
        cntr += len(d)
    # 统计所有的特征值
    u_features = np.zeros((num_users, cntr), dtype=np.float32)
    for _, row in users_df.iterrows():
        u_id = row['user_id']
        # 确保该用户有电影的评分记录
        if u_id in u_dict.keys():
            for k, header in enumerate(cols):
                u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.
    u_features = sparse.csr_matrix(u_features)
    v_features = sparse.csr_matrix(v_features)

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items)))

    class_values = np.sort(np.unique(ratings))
    return num_users, num_items, u_features, v_features, class_values,\
        ratings_train, ratings_val, ratings_test


def get_data(seed=1234, verbose=True):
    """加载数据集
    """
    num_users, num_items, u_features, v_features, class_values, \
        ratings_train, ratings_val, ratings_test = load_data(seed=seed, verbose=verbose)

    print("Normalizing feature vectors...")
    u_features_side = normalize_features(u_features)
    v_features_side = normalize_features(v_features)

    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

    # 943x41, 1682x41
    u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
    v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

    num_side_features = u_features_side.shape[1]

    # node id's for node input features
    id_csr_u = sparse.identity(num_users, format='csr')
    id_csr_v = sparse.identity(num_items, format='csr')

    u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

    u_features = u_features.toarray()
    v_features = v_features.toarray()

    num_features = u_features.shape[1]

    return num_users, num_items, len(class_values), num_side_features, num_features, \
        u_features, v_features, u_features_side, v_features_side, \
            ratings_train, ratings_val, ratings_test
        

if __name__ == '__main__':
    print("--------------------------------")
    print(get_data())
