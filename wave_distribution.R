library(ggplot2)
library(ggridges)
library(tidyr)

bar_t2ds_phenodata_list <- read.csv('./data/stat_data/bar_t2ds_phenodata_list.csv')
bar_no_t2ds_phenodata_list <- read.csv('./data/stat_data/bar_no_t2ds_phenodata_list.csv')
bar_pret2ds_phenodata_list <- read.csv('./data/stat_data/bar_pret2ds_phenodata_list.csv')

# Add a column to identify each dataset
bar_t2ds_phenodata_list$dataset <- 'T2DS'
bar_no_t2ds_phenodata_list$dataset <- 'No T2DS'
bar_pret2ds_phenodata_list$dataset <- 'Pre-T2DS'

# Combine all datasets into one dataframe
combined_data <- rbind(bar_t2ds_phenodata_list, bar_no_t2ds_phenodata_list, bar_pret2ds_phenodata_list)

# Reshape data to long format for ggplot2
long_data <- pivot_longer(combined_data, 
                          cols = -dataset,       # All columns except the 'dataset' column
                          names_to = "feature",  
                          values_to = "value")   

ggsave("wave_distribution_plot.png", 
       ggplot(long_data, aes(x = value, y = feature, fill = dataset, color = dataset)) + 
         geom_density_ridges(alpha = 0.2, scale = 1, rel_min_height = 0.01) +  
         scale_fill_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c")) +      
         scale_color_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c")) +     
         theme_ridges() +                                                      
         theme(
           panel.background = element_rect(fill = "white", colour = "white"),   
           plot.background = element_rect(fill = "white", colour = "white"),    
           panel.grid.major = element_line(colour = "grey90"),                  
           panel.grid.minor = element_blank(),                                  
           axis.text = element_text(colour = "black"),                          
           axis.title = element_text(colour = "black"),                        
           plot.title = element_text(hjust = 0.5, colour = "black")            
         ) +
         labs(title = "Wave Distribution of Features with Different Datasets",  
              x = "Value Distribution",                                       
              y = "Features") +                                               
         xlim(-1, 1)                                                          
       
)
