library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(Jmisc)

setwd('C:/Users/hemingzhang/Documents/vs-files/LLFS-GiG')
type = 't2ds'
type_path = paste('./analysis/gigtransformer/', as.character(type), sep='')
refilter_node_path = paste(type_path, '_refilter_node_weight_df.csv', sep='')
sorted_refilter_net_node = read.csv(refilter_node_path)

subject_nodeidx_gene_df = read.csv('./data/filtered_data/merged_tran_v1_nodeidx_df.csv')
label_patient_nodeidx_df = read.csv('./data/filtered_data/label_phenodata_onehot_nodeidx_df.csv')
t2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$t2ds == 1, ]
pret2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$pret2ds == 1, ]
no_t2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$no_t2ds == 1, ]

t2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% t2ds_nodeidx_df$node_idx, ]
pret2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% pret2ds_nodeidx_df$node_idx, ]
no_t2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% no_t2ds_nodeidx_df$node_idx, ]

for(i in 1:nrow(sorted_refilter_net_node)) {
  gene_name <- sorted_refilter_net_node[i, 'gene_node_name']
  t2ds_gene_value_list <- t2ds_subject_nodeidx_gene_df[[gene_name]]
  pret2ds_gene_value_list <- pret2ds_subject_nodeidx_gene_df[[gene_name]]
  no_t2ds_gene_value_list <- no_t2ds_subject_nodeidx_gene_df[[gene_name]]
  
  t2ds_pret2ds_test_result <- wilcox.test(t2ds_gene_value_list, pret2ds_gene_value_list)
  print(t2ds_pret2ds_test_result$p.value)
  sorted_refilter_net_node$t2ds_pret2ds_pvalue[i] = t2ds_pret2ds_test_result$p.value
  
  t2ds_no_t2ds_test_result <- wilcox.test(t2ds_gene_value_list, no_t2ds_gene_value_list)
  print(t2ds_no_t2ds_test_result$p.value)
  sorted_refilter_net_node$t2ds_no_t2ds_pvalue[i] = t2ds_no_t2ds_test_result$p.value
  
  pret2ds_no_t2ds_test_result <- wilcox.test(pret2ds_gene_value_list, no_t2ds_gene_value_list)
  print(pret2ds_no_t2ds_test_result$p.value)
  sorted_refilter_net_node$pret2ds_no_t2ds_pvalue[i] = pret2ds_no_t2ds_test_result$p.value
}


gene_num_dict = read.csv('./data/filtered_data/gene_num_dict_df.csv')

for(i in 1:nrow(gene_num_dict)) {
  gene_name <- gene_num_dict[i, 'gene_node_name']
  if (grepl('-', gene_name) == TRUE){
    gene_name <- gsub('-', '.', gene_name)
  }
  t2ds_gene_value_list <- t2ds_subject_nodeidx_gene_df[[gene_name]]
  print(t2ds_gene_value_list)
  pret2ds_gene_value_list <- pret2ds_subject_nodeidx_gene_df[[gene_name]]
  no_t2ds_gene_value_list <- no_t2ds_subject_nodeidx_gene_df[[gene_name]]
  
  t2ds_pret2ds_test_result <- wilcox.test(t2ds_gene_value_list, pret2ds_gene_value_list)
  print(t2ds_pret2ds_test_result$p.value)
  gene_num_dict$t2ds_pret2ds_pvalue[i] = t2ds_pret2ds_test_result$p.value
  
  t2ds_no_t2ds_test_result <- wilcox.test(t2ds_gene_value_list, no_t2ds_gene_value_list)
  print(t2ds_no_t2ds_test_result$p.value)
  gene_num_dict$t2ds_no_t2ds_pvalue[i] = t2ds_no_t2ds_test_result$p.value
  
  pret2ds_no_t2ds_test_result <- wilcox.test(pret2ds_gene_value_list, no_t2ds_gene_value_list)
  print(pret2ds_no_t2ds_test_result$p.value)
  gene_num_dict$pret2ds_no_t2ds_pvalue[i] = pret2ds_no_t2ds_test_result$p.value
}


pvalue = 0.2
pvalue_sorted_refilter_net_node <- filter(sorted_refilter_net_node, t2ds_no_t2ds_pvalue <= pvalue)
pvalue_gene_num_dict = filter(gene_num_dict, t2ds_no_t2ds_pvalue <= pvalue)





