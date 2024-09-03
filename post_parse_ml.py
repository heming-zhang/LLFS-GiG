import os
import numpy as np
import pandas as pd

class LabelParse():
    def __init__(self):
        pass
    
    # Randomize the [label]
    def shuffle_split(self):
        label_df = pd.read_csv('./data/filtered_data/label_phenodata_onehot_nodeidx_df.csv')
        random_label_df = label_df.sample(frac = 1)
        random_label_df.to_csv('./data/filtered_data/random_label_phenodata_onehot_nodeidx_df.csv', header=True, index=False)

    # Split the label input into [Training / Test dataset]
    def split_k_fold(self, k):
        random_label_df = pd.read_csv('./data/filtered_data/random_label_phenodata_onehot_nodeidx_df.csv')
        num_points = random_label_df.shape[0]

        num_div = int(num_points / k)
        num_div_list = [i * num_div for i in range(0, k)]
        num_div_list.append(num_points)
        # SPLIT [random_label_df] INTO [k] FOLDS
        for place_num in range(k):
            low_idx = num_div_list[place_num]
            high_idx = num_div_list[place_num + 1]
            # print(low_idx, high_idx)
            print('\n--------TRAIN-TEST SPLIT WITH TEST FROM ' + str(low_idx) + ' TO ' + str(high_idx) + '--------')
            split_label_df = random_label_df[low_idx : high_idx]
            print(split_label_df.shape)
            split_label_df.to_csv('./data/filtered_data/random_label_phenodata_onehot_nodeidx_df_' + str(place_num + 1) + '.csv', index=False, header=True)
    
    # GENERATE [fold_n] [train_label]
    def train_test(self, fold_n, k):
        train_label_dflist = []
        for i in range(1, k + 1):
            if i == fold_n:
                print('--- LOADING ' + str(i) + '-TH SPLIT TEST DATA ---')
                test_label_df = pd.read_csv('./data/filtered_data/random_label_phenodata_onehot_nodeidx_df_' + str(i) + '.csv')
                test_label_df.to_csv('./data/filtered_data/test_label_' + str(fold_n) + '.csv', index=False, header=True)
            else:
                print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
                train_label_df = pd.read_csv('./data/filtered_data/random_label_phenodata_onehot_nodeidx_df_' + str(i) + '.csv')
                train_label_dflist.append(train_label_df)
        print('--- COMBINING DATA ... ---')
        train_label_df = pd.concat(train_label_dflist)
        print(train_label_df)
        train_label_df.to_csv('./data/filtered_data/train_label_' + str(fold_n) + '.csv', index=False, header=True)


class LoadData():
    def __init__(self):
        pass

    def load_data(self, fold_n):
        subfeature_dict_df = pd.read_csv('./data/filtered_data/subfeature_dict_df.csv')
        num_subfeature = subfeature_dict_df.shape[0]
        # [train_label]
        train_label_df = pd.read_csv('./data/filtered_data/train_label_' + str(fold_n) + '.csv')
        train_idx = np.array(train_label_df['node_idx'])
        train_idx = train_idx - num_subfeature
        np.save('./data/post_ml_data/train_idx.npy', train_idx)
        print(train_idx)
        train_label = train_label_df.drop(columns=['node_idx']).to_numpy()
        print(train_label)
        np.save('./data/post_ml_data/train_label.npy', train_label)
        print(train_label.shape)
        # [train_x]
        x = np.load('./data/post_ml_data/x.npy')
        train_x = x[train_idx]
        np.save('./data/post_ml_data/train_x.npy', train_x)
        print(train_x.shape)
        # [train_norm_x]
        norm_x = np.load('./data/post_ml_data/norm_x.npy')
        train_norm_x = norm_x[train_idx]
        np.save('./data/post_ml_data/train_norm_x.npy', train_norm_x)
        print(train_norm_x.shape)

        # [test_label]
        test_label_df = pd.read_csv('./data/filtered_data/test_label_' + str(fold_n) + '.csv')
        test_idx = np.array(test_label_df['node_idx'])
        test_idx = test_idx - num_subfeature
        np.save('./data/post_ml_data/test_idx.npy', test_idx)
        print(test_idx)
        test_label = test_label_df.drop(columns=['node_idx']).to_numpy()
        print(test_label)
        np.save('./data/post_ml_data/test_label.npy', test_label)
        print(test_label.shape)
        # [test_x]
        test_x = x[test_idx]
        np.save('./data/post_ml_data/test_x.npy', test_x)
        print(test_x.shape)
        # [test_norm_x]
        test_norm_x = norm_x[test_idx]
        np.save('./data/post_ml_data/test_norm_x.npy', test_norm_x)
        print(test_norm_x.shape)


if __name__ == "__main__":
    ### K-fold parameters
    k = 5
    fold_n = 1

    ### Formalize label with randomization
    LabelParse().shuffle_split()
    LabelParse().split_k_fold(k)

    ### K-FOLD SPLIT
    LabelParse().train_test(fold_n, k)

    ### FORMALZE [torch geometric] DATA
    LoadData().load_data(fold_n)