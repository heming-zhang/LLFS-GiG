import os
import pdb
import torch
import shutil
import argparse
import tensorboardX
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR

import utils
from geo_loader.geo_readgraph import read_geodata
from post_parse import LabelParse, LoadGeoData
from gnn_acc_analysis import acc_f1_performance
from enc.geo_gigtransformer_analysis import GIG_Transformer

from post_parse import LabelParse, LoadGeoData


def build_geogig_model(args, num_gene_node, device):
    print('--- BUILDING UP GNN MODEL ... ---')
    # Get parameters
    model = GIG_Transformer(gene_input_dim=args.gene_input_dim, 
                gene_hidden_dim=args.gene_hidden_dim, 
                gene_embedding_dim=args.gene_output_dim,
                gene_num_top_feature=args.gene_num_top_feature,
                num_gene_node=num_gene_node,
                gig_input_dim=args.gig_input_dim,
                gig_input_transform_dim=args.gig_input_transform_dim,
                gig_hidden_dim=args.gig_hidden_dim, 
                gig_embedding_dim=args.gig_output_dim,
                num_classes=args.num_classes, 
                gene_num_head=args.gene_num_head,
                gig_num_head=args.gig_num_head,
                class_weight_fine=args.class_weight_fine,
                class_weight=args.class_weight,
                ortho_weight=args.ortho_weight,
                link_weight=args.link_weight,
                ent_weight=args.ent_weight,
                graph_opt=args.graph_opt)
    model = model.to(device)
    return model


def test_geogig_model(fold_n, data, num_feature, num_subfeature, num_subject, num_gene_node,
                        gene_feature, gene_edge_index, model, device, args):
    loss = 0
    x = Variable(data.x, requires_grad=False).to(device)
    edge_index = Variable(data.edge_index, requires_grad=False).to(device)
    node_label = Variable(data.node_label, requires_grad=False).to(device)
    node_index = Variable(data.node_index, requires_grad=False).to(device)
    x_embed, node_output, ypred, y_nodepred = model(fold_n, num_feature=num_feature, num_subfeature=num_subfeature, 
                                                    num_subject=num_subject, num_gene_node=num_gene_node, 
                                                    gene_feature=gene_feature, gene_edge_index=gene_edge_index,
                                                    x=x, edge_index=edge_index, 
                                                    node_label=node_label, node_index=node_index,
                                                    args=args, device=device)
    loss = model.loss(node_output, node_label, gene_edge_index, args.gene_num_top_feature)
    loss = loss.item()
    return model, loss, x_embed, node_output, ypred, y_nodepred


def test_geogig(fold_n, model, device, args):
    # Test model on training dataset with [gene features]
    num_feature = 1
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]
    gene_feature = np.load('./data/post_data/gene_x.npy')
    gene_edge_index = np.load('./data/post_data/gene_edge_index.npy')
    gene_edge_index = torch.from_numpy(gene_edge_index).long()

    # Test model on training dataset with [subject features]
    subfeature_dict_df = pd.read_csv('./data/filtered_data/subfeature_dict_df.csv')
    num_subfeature = subfeature_dict_df.shape[0]
    subject_dict_df = pd.read_csv('./data/filtered_data/subject_dict_df.csv')
    num_subject = subject_dict_df.shape[0]
    graph_feature = np.load('./data/post_data/x.npy')
    edge_index = np.load('./data/post_data/edge_index.npy')
    node_label = np.load('./data/post_data/test_label.npy')
    node_label_indices = np.argmax(node_label, axis=1)
    node_idx = np.load('./data/post_data/test_idx.npy')

    # Run test model
    model.eval()
    geo_data = read_geodata(graph_feature, edge_index, node_label, node_idx)
    print('TEST MODEL ...')
    model, test_loss, x_embed, node_output, ypred, y_nodepred = test_geogig_model(fold_n, geo_data, 
                                                                num_feature, num_subfeature, num_subject, num_gene_node,
                                                                gene_feature, gene_edge_index,
                                                                model, device, args)
    print('TEST LOSS: ', test_loss)
    y_nodepred = y_nodepred.cpu().detach().numpy()
    test_correct_count = (y_nodepred == node_label_indices).sum()
    test_accuracy = float(test_correct_count) / len(node_label_indices)
    print('TEST CORRECT: ', test_correct_count)
    print('TEST ACCURACY: ', test_accuracy)
    # Compute confusion matrix
    test_confusion_matrix = confusion_matrix(node_label_indices, y_nodepred)
    print(test_confusion_matrix)
    # Create [test_label_df] for saving preparation
    test_label_df = pd.DataFrame({'test_node_idx': list(node_idx),
                                'test_label': list(node_label_indices),
                                'test_pred_label': list(y_nodepred)})
    return test_accuracy, test_confusion_matrix, test_label_df, test_loss

# Parse arguments from command line
def arg_parse():
    parser = argparse.ArgumentParser(description='GEO-WEBGNN ARGUMENTS.')
    # Set default input argument
    parser.set_defaults(cuda = '0',
                        parallel = False,
                        add_self = '0', # 'add'
                        adj = '0', # 'sym'
                        model = '0', # 'load'
                        lr = 0.01,
                        # lr = 0.075,
                        weight_decay = 1e-10,
                        milestones = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250],
                        gamma = 0.9,
                        clip = 5.0,
                        batch_size = 256,
                        num_epochs = 500,
                        unchanged_threshold = 100,
                        change_wave = 0.75,
                        num_workers = 0,
                        # graph_opt = 'gene',
                        # graph_opt = 'subject',
                        graph_opt = 'GinG',
                        gene_input_dim = 6, # gene embedding parameters
                        gene_hidden_dim = 18,
                        gene_output_dim = 18,
                        gene_num_top_feature = 18,
                        gig_input_dim = 42, # gig embedding parameters
                        gig_input_transform_dim = 18,
                        gig_hidden_dim = 18,
                        gig_output_dim = 18,
                        class_weight_fine = 0.5,
                        class_weight = 0.9, # loss function parameters
                        ortho_weight = 0.05,
                        link_weight = 0.05,
                        ent_weight = 0.00,
                        num_classes = 3,
                        gene_num_head = 1,
                        gig_num_head = 1,
                        dropout = 0.01)
    return parser.parse_args()


def analysis_model(k, fold_n, args):
    ### Prepare the hyperparameters
    prog_args = arg_parse() # Parse argument from terminal or default parameters
    device, prog_args.gpu_ids = utils.get_available_devices() # Check and allocate resources
    device = torch.device('cuda:0') # Manual set
    torch.cuda.set_device(device)
    prog_args.gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('MAIN DEVICE: ', device)

    ### Prepare the dataset with [k-fold cross validation]
    LabelParse().train_test(fold_n, k) # Formalize label
    LoadGeoData().geo_load_data(fold_n) # Formalize [torch geometric] data

    # Build [GinG Transformer] Model
    test_load_path = './gnn_result/gigtransformer/5-fold/epoch_' + str(args.num_epochs) + '_fold' + str(fold_n) + '/best_train_model.pth'
    test_save_path = './gnn_result/gigtransformer/5-fold/epoch_' + str(args.num_epochs) + '_fold' + str(fold_n)
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]
    model = build_geogig_model(prog_args, num_gene_node, device)
    model.load_state_dict(torch.load(test_load_path, map_location=device))
    test_accuracy, test_confusion_matrix, test_label_df, test_loss = test_geogig(fold_n, model, device, prog_args)


if __name__ == "__main__":
    k = 5
    fold_n = 1
    analysis_model(k, fold_n)
