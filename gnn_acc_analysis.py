import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, f1_score



def acc_f1_performance(fold_n, path):
    print('-------------------------- TRAINING METRICS -----------------------------')
    train_label_df = pd.read_csv(path + '/result_train_label_' + str(fold_n) + '.csv')
    train_label = np.array(list(train_label_df['train_label']))
    y_nodepred = np.array(list(train_label_df['train_pred_label']))
    train_correct_count = (y_nodepred == train_label).sum()
    train_accuracy = float(train_correct_count) / len(train_label)
    print('TRAIN CORRECT: ', train_correct_count)
    print('TRAIN ACCURACY: ', train_accuracy)
    # Compute confusion matrix
    train_confusion_matrix = confusion_matrix(train_label, y_nodepred)
    print(train_confusion_matrix)
    train_TP = np.diag(train_confusion_matrix)
    train_FN = np.sum(train_confusion_matrix, axis=0) - train_TP
    train_FP = np.sum(train_confusion_matrix, axis=1) - train_TP
    train_TN = np.sum(train_confusion_matrix) - (train_TP + train_FP + train_FN)
    print('TP: ', train_TP)
    print('FN: ', train_FN)
    print('FP: ', train_FP)
    print('TN: ', train_TN)
    # Calculate [TPR FPR F1]
    # Calculate true positive rate (TPR) or recall for each class
    train_TPR = np.nan_to_num(train_TP / (train_TP + train_FN))
    # Calculate true negative rate (TNR) or specificity for each class
    train_TNR = np.nan_to_num(train_TN / (train_TN + train_FP))
    # Calculate F1 score for each class
    precision = np.nan_to_num(train_TP / (train_TP + train_FP))
    recall = np.nan_to_num(train_TP / (train_TP + train_FN))
    train_F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    train_micro_F1 = np.mean(train_F1)
    print('TRAIN TPR/Sensitivity/Recall: ', train_TPR)
    print('TRAIN FPR/Specificity/Selectivity: ', train_TNR)
    print('TRAIN F1 Score (Micro-averaged): ', train_micro_F1)

    print('-------------------------- TEST METRICS -----------------------------')
    test_label_df = pd.read_csv(path + '/result_test_label_' + str(fold_n) + '.csv')
    test_label = np.array(list(test_label_df['test_label']))
    y_nodepred = np.array(list(test_label_df['test_pred_label']))
    test_correct_count = (y_nodepred == test_label).sum()
    test_accuracy = float(test_correct_count) / len(test_label)
    print('TEST CORRECT: ', test_correct_count)
    print('TEST ACCURACY: ', test_accuracy)
    # Compute confusion matrix
    test_confusion_matrix = confusion_matrix(test_label, y_nodepred)
    print(test_confusion_matrix)
    test_TP = np.diag(test_confusion_matrix)
    test_FN = np.sum(test_confusion_matrix, axis=0) - test_TP
    test_FP = np.sum(test_confusion_matrix, axis=1) - test_TP
    test_TN = np.sum(test_confusion_matrix) - (test_TP + test_FP + test_FN)
    print('TP: ', test_TP)
    print('FN: ', test_FN)
    print('FP: ', test_FP)
    print('TN: ', test_TN)
    # Calculate [TPR FPR F1]
    # Calculate true positive rate (TPR) or recall for each class
    test_TPR = np.nan_to_num(test_TP / (test_TP + test_FN))
    # Calculate true negative rate (TNR) or specificity for each class
    test_TNR = np.nan_to_num(test_TN / (test_TN + test_FP))
    # Calculate F1 score for each class
    precision = np.nan_to_num(test_TP / (test_TP + test_FP))
    recall = np.nan_to_num(test_TP / (test_TP + test_FN))
    test_F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    test_micro_F1 = np.mean(test_F1)
    print('TEST TPR/Sensitivity/Recall: ', test_TPR)
    print('TEST FPR/Specificity/Selectivity: ', test_TNR)
    print('TEST F1 Score (Micro-averaged): ', test_micro_F1)
    return train_confusion_matrix, train_accuracy, test_confusion_matrix, test_accuracy

def summarize_bestunit(number_iteration):
    bestunit_list = []
    max_test_accuracy_list = []
    num_epochs = 1500
    fold_train_test_dict = {'fold1-train': [], 'fold1-test': [], 'fold2-train': [], 'fold2-test': [], 'fold3-train': [], 'fold3-test': [], 'fold4-train': [], 'fold4-test': [], 'fold5-train': [], 'fold5-test': []}
    for fold_n in range(1, 2):
        fold_trainacc_list = []
        fold_testacc_list = []
        max_test_accuracy = 0
        best_unit = 0
        for unit in range(1, number_iteration + 1):
            fold_n_best_unit_folder_path = './gnn_result/epoch_' + str(num_epochs) + '_fold' + str(fold_n) + '-' + str(unit)
            train_confusion_matrix, train_accuracy, test_confusion_matrix, test_accuracy = acc_f1_performance(fold_n, fold_n_best_unit_folder_path)
            fold_trainacc_list.append(train_accuracy)
            fold_testacc_list.append(test_accuracy)
            if test_accuracy > max_test_accuracy:
                max_test_accuracy = test_accuracy
                best_unit = unit
        bestunit_list.append(best_unit)
        max_test_accuracy_list.append(max_test_accuracy)
        print('Fold ', fold_n, ' Best Unit: ', best_unit)
        fold_train_test_dict['fold' + str(fold_n) + '-train'] = fold_trainacc_list
        fold_train_test_dict['fold' + str(fold_n) + '-test'] = fold_testacc_list
        print(fold_trainacc_list)
        print(fold_testacc_list)
    print('Best Unit List: ', bestunit_list)
    print('Max Test Accuracy List: ', max_test_accuracy_list)
    # Convert to dataframe
    fold_train_test_df = pd.DataFrame(fold_train_test_dict)
    print(fold_train_test_df)

if __name__ == "__main__":
    ## Summarize the best performance of each fold
    number_iteration = 10
    summarize_bestunit(number_iteration)

    # ### Check specific fold and Unit with unknown max test accuracy
    # num_epochs = 1500
    # fold_n = 3
    # unit = 2
    # fold_n_best_unit_folder_path = './gnn_result/epoch_' + str(num_epochs) + '_fold' + str(fold_n) + '-' + str(unit)
    # fold_n_best_folder_path = './gnn_result/epoch_' + str(num_epochs) + '_fold' + str(fold_n) + '-best'
    # acc_f1_performance(fold_n, fold_n_best_folder_path)

    # ### Check specific fold and Unit with known max test accuracy
    # model_name = 'gigtran-pheno'
    # # model_name = 'gigtran-gene'
    # # model_name = 'gigtran'
    # fold_n = 3
    # fold_n_saved_folder_path = './gnn_result/' + model_name + '/5-fold/fold_' + str(fold_n)
    # acc_f1_performance(fold_n, fold_n_saved_folder_path)