# Longevity-GiG

## 1. Data Processing
### 1.1 Multi-omics Features
* tran_v1_df ->(ensembl_data)-> merged_tran_v1_df -> filter (all_edge_gene_list) -> merged_tran_v1_df
* [core_promoter/distal_promoter/proximal_promoter/downstream/upstream]_df ->(ensembl_data)-> merged_[]_df -> filter (all_edge_gene_list) -> merged_[]_df
* Overlapping of (merged_tran_v1_df & merged_[]_df ) -> intersected_gene 
* kegg_df -> up_kegg_df -> Overlapping nodes -> all_edge_gene_list


### 1.2 Clinical Features / Labels
* phenodata_df -> subject_list
* t2ds_label_df -> t2ds_label_subject_list
* merged_tran_v1_df -> merged_tran_v1_subject_list
* merged_[]_df -> merged_[upstream]_subject_list
* Overlapping of (subject_list & t2ds_label_subject_list & merged_tran_v1_subject_list & merged_upstream_subject_list) -> intersected_subject_list 

Hence,
* phenodata_df -> filter (intersected_subject_list) -> phenodata_df
* t2ds_label_df -> filter (intersected_subject_list) -> t2ds_label_df
* merged_tran_v1_df -> filter (intersected_subject_list) -> merged_tran_v1_df
* merged_[]_df -> filter (intersected_subject_list) -> merged_[]_df

Cleaning clinical data
* phenodata_df -> label_phenodata_df ->  (v1) -> v1_label_phenodata_df

## 2. Graph Construction
### 2.1 Patient Graph Nodes
* v1_label_phenodata_df -> (onehot encoding) -> v1_label_phenodata_onehot_df
* v1_label_phenodata_onehot_df -> (10/90 percentile categorize) -> v1_label_phenodata_category_df -> subfeature_dict_df

* v1_label_phenodata_category_df -> subfeature_dict_df (phenotypes mapping)
* (subject_list / subject_name_list) -> subject_dict_df (subject mapping) -> subject_number_dict_df
* concatenate [subfeature_dict_df, subject_number_dict_df] ->  node_idx_name_map_df (all nodes mapping)

### 2.2 Patient Graph Edges
* v1_label_phenodata_category_df -> (replacing categorized values into node num) -> v1_label_phenodata_category_name_df -> mapping with (node_idx_name_map_df) -> v1_label_phenodata_category_num_df -> v1_label_phenodata_category_num_dflist -> v1_label_phenodata_category_edge_df (this is part for constructing connections between patient subjects and corresponding categorized features)
* v1_label_phenodata_category_df / v1_label_phenodata_feature_list -> subfeature_name_edge_df -> (node_name_idx_dict) -> subfeature_num_edge_df

* concatenate [subfeature_num_edge_df, v1_label_phenodata_category_edge_df] -> num_edge_df -> edge_index

### 2.3 Patient Graph Features
* v1_label_phenodata_onehot_df ->(subject_node_dict)-> v1_label_phenodata_onehot_nodeidx_df
* v1_label_phenodata_onehot_nodeidx_df -> x_v1_label_phenodata_onehot_nodeidx_df -> subject_phenodata_x
* np.zeros((num_subfeature, num_feature)) -> subfeature_phenodata_x
* concatenate [subfeature_phenodata_x, subject_phenodata_x] -> x

### 2.4 Gene Graph Multi-omics Feautures
* merged_tran_v1_df ->(subject_node_dict)-> merged_tran_v1_nodeidx_df
* merged_[]_df ->(subject_node_dict)->  merged_[]_nodeidx_df

### 2.5 Gene Graph Nodes
* merged_tran_v1_df -> gene_name_list -> gene_node_idx_list -> gene_name_dict -> gene_num_dict_df

### 2.6 Gene Graph Edges
* up_kegg_df ->(gene_name_dict)-> gene_num_edge_df -> reverse_gene_num_edge_df
                               -> gene_edge_index

### 2.7 Gene Graph Features
* merged_tran_v1_nodeidx_df -> gene_tran_x_df -> gene_tran_x -> norm_gene_tran_x
* merged_[]_nodeidx_df -> gene_[]_x_df -> gene_[]_x -> norm_[]_x
* concatenate [gene_tran_x, gene_core_promoter_x, gene_proximal_promoter_x, gene_distal_promoter_x, gene_upstream_x, gene_downstream_x] -> gene_x
* concatenate [norm_gene_tran_x, norm_gene_core_promoter_x, norm_gene_proximal_promoter_x, norm_gene_distal_promoter_x, norm_gene_upstream_x, norm_gene_downstream_x] -> norm_gene_x

## 3. Common GNN Graph Construction
### 3.1 Graph Node Features
* v1_label_phenodata_onehot_df -> pheno_x_df -> pheno_x -> norm_pheno_x
* geno_pheno_x = np.hstack((gene_x, pheno_x)) 
* fill np.zeros((num_subfeature, dim_geno_pheno_x)) -> subfeature_all_x
* all_x = np.vstack((subfeature_all_x, geno_pheno_x))   [935, 8340+42]

* norm_geno_pheno_x = np.hstack((norm_gene_x, norm_pheno_x))
* fill np.zeros((num_subfeature, dim_norm_geno_pheno_x)) -> norm_subfeature_all_x
* norm_all_x = np.vstack((norm_subfeature_all_x, norm_geno_pheno_x))

### 3.2 Graph Edge Features
* Can Just USE: num_edge_df -> edge_index





