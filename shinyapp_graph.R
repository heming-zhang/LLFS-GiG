library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(Jmisc)


ui <- fluidPage(
  # titlePanel('Whole Network Interaction'),
  sidebarLayout(
    sidebarPanel(
      selectInput('diabetes_type', label = 'Selection of the type of diabetes',
                  choices = list('T2ds' = 1, 
                                 'Pret2ds' = 2, 
                                 'No_t2ds' = 3),
                  selected = 1),
      
      sliderInput('edge_threshold',
                  'Select the threshold of edge weight to plot',
                  min = 0.1, max = 1.0,
                  value = 0.95),
      
      sliderInput('node_threshold',
                  'Select the threshold of marking important genes',
                  min = 0, max = 1.0,
                  value = 0.75),
      
      sliderInput('giant_comp_threshold',
                  'Select the threshold of each component',
                  min = 0.0, max = 50.0,
                  value = 10.0),

      sliderInput('gene_node_size',
                  'Select the gene node size',
                  min = 0, max = 10,
                  value = 4.0),

      sliderInput('imgene_node_size',
                  'Select the important gene node size',
                  min = 0, max = 10,
                  value = 7.0),

      sliderInput('gene_label_size',
                  'Select the label size of gene nodes',
                  min = 0.2, max = 1.0,
                  value = 0.6),

      sliderInput('imgene_label_size',
                  'Select the label size of important genes',
                  min = 0.4, max = 1.5,
                  value = 0.8),
    ),
    mainPanel(
      plotOutput(outputId = 'network', height = 1150, width = 1150)
    )
  )
)

server <- function(input, output) {
  edge_threshold <- reactive({
    input$edge_threshold
  })
  node_threshold <- reactive({
    input$node_threshold
  })
  giant_comp_threshold <- reactive({
    input$giant_comp_threshold
  })
  output$network <- renderPlot({
    ### 1. READ GRAPH [edge_index, node] FROM FILES
    print(input$diabetes_type)
    if (input$diabetes_type == 1){
      type = 't2ds'
    } else if (input$diabetes_type == 2){
      type = 'pret2ds'
    } else if (input$diabetes_type == 3){
      type = 'no_t2ds'
    }
    type_path = paste('./analysis/gigtransformer/', as.character(type), sep='')
    edge_path = paste(type_path, '_norm_layer_average_fold_gene_edge_weight_df.csv', sep='')
    net_edge_weight = read.csv(edge_path)
    all_net_node = read.csv('./data/filtered_data/gene_num_dict_df.csv') # NODE LABEL
    node_path = paste(type_path, '_layer_average_fold_node_weight_df.csv', sep='')
    type_net_node = read.csv('./analysis/gigtransformer/t2ds_layer_average_fold_node_weight_df.csv')
    net_node = merge(x = all_net_node, y = type_net_node, by.x = c('gene_node_idx'), by.y =c('Node_idx'))
    
    # ### 2.1 FILTER EDGE BY [edge_weight]
    # filter_net_edge = filter(net_edge_weight, Weight > edge_threshold())
    # filter_net_edge_node = unique(c(filter_net_edge$Actual_From, filter_net_edge$Actual_To))
    # filter_net_node = net_node[net_node$gene_node_idx %in% filter_net_edge_node, ]
    
    ### 2.1 FILTER EDGE BY [node_weight]
    filter_net_node = filter(net_node, Weight > node_threshold())
    filter_net_edge = net_edge_weight[net_edge_weight$Actual_From %in% filter_net_node$gene_node_idx & net_edge_weight$Actual_To %in% filter_net_node$gene_node_idx, ]
    print(filter_net_node)
    print(filter_net_edge)
    
    
    ### 2.2 FILTER WITH GIANT COMPONENT
    tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
    # tmp_net = graph_from_data_frame(d=hop_net_edge, vertices=net_node, directed=F)
    all_components = groups(components(tmp_net))
    # COLLECT ALL LARGE COMPONENTS
    giant_comp_node = c()
    for (x in 1:length(all_components)){
      each_comp = all_components[[x]]
      if (length(each_comp) >= giant_comp_threshold()){
        giant_comp_node = c(giant_comp_node, each_comp)
      }
    }
    
    refilter_net_edge<-subset(filter_net_edge, (Actual_From %in% giant_comp_node | Actual_To %in% giant_comp_node))
    refilter_net_edge_node = unique(c(refilter_net_edge$Actual_From, refilter_net_edge$Actual_To))
    refilter_net_node = filter_net_node[filter_net_node$gene_node_idx %in% refilter_net_edge_node,]
    
    
    # 3.2 USE REWEIGHTED IDF NODE DEGREE AS NODE DEGREE
    print(nrow(refilter_net_edge))
    print(nrow(refilter_net_node))
    sorted_refilter_net_node <- refilter_net_node[order(refilter_net_node$Weight, decreasing = TRUE), ]
    refilter_edge_path = paste(type_path, '_refilter_edge_weight_df.csv', sep='')
    write.csv(refilter_net_edge, refilter_edge_path)
    refilter_node_path = paste(type_path, '_refilter_node_weight_df.csv', sep='')
    write.csv(sorted_refilter_net_node, refilter_node_path)
    
    net = graph_from_data_frame(d=refilter_net_edge, vertices=refilter_net_node, directed=F)
    
    ### 4. NETWORK PARAMETERS SETTINGS
    # vertex frame color
    vertex_fcol = rep('black', vcount(net))
    # vertex color
    vertex_col = rep('lightblue', vcount(net))
    vertex_col[V(net)$Weight>=node_threshold()] = 'lightblue' # 'tomato'
    # vertex size
    vertex_size = rep(input$gene_node_size, vcount(net))
    vertex_size[V(net)$Weight>=node_threshold()] = input$imgene_node_size
    # vertex cex
    vertex_cex = rep(input$gene_label_size, vcount(net))
    vertex_cex[V(net)$Weight>=node_threshold()] = input$imgene_label_size
    # edge width
    edge_width = (E(net)$Weight)*(1.0)
    edge_width[E(net)$Weight>=edge_threshold()] = (E(net)$Weight)*(5.0)
    # edge color
    edge_color = rep('gray', ecount(net))
    edge_color[E(net)$Weight>=edge_threshold()] = 'black'
    
    set.seed(18)
    plot(net,
         vertex.frame.width = 2,
         vertex.frame.color = vertex_fcol,
         vertex.color = vertex_col,
         vertex.size = vertex_size,
         vertex.label = V(net)$gene_node_name,
         vertex.label.color = 'black',
         vertex.label.cex = vertex_cex,
         edge.width = edge_width,
         edge.color = edge_color,
         edge.curved = 0.2,
         layout=layout_nicely)
    ### ADD LEGEND
    legend(x=-1.05, y=1.10, # y= -0.72,
           legend=c('Genes'), pch=c(21), 
           pt.bg=c('lightblue'), pt.cex=2, cex=1.2, bty='n')
    legend(x=-1.06, y=1.05, # y= -0.85, 
           legend=c('Gene-Gene', 'Important Gene-Gene'),
           col=c('gray', 'black'), lwd=c(5,7), cex=1.2, bty='n')
  })
}

# layout=layout_with_graphopt
# layout=layout_with_sugiyama
# layout=layout_with_lgl
# layout = layout.random
# layout=layout_nicely
# layout=layout_as_tree
# layout_with_kk
# layout=layout_with_dh
# layout=layout_with_gem

shinyApp(ui = ui, server = server)


