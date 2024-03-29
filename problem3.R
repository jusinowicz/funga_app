#=============================================================================
# Instructions:
#  3. Assume climate change will decrease soil pH across the dataset. Provide a
#  recommendation for sites that will be productive planting locations under this scenario.
#
# This will build off of problem 2, and also help answer the last part of problem
# 2. The simplest thing is to follow the same approach as with setting stand
# age to 25, and just set the pH (phh2o) to the minimum value in the data set, 
# then generate predictions. 
#
# Also, MEASYEAR is not going to be useful here.  
#
# Datasets: 
#	tree_level_lob_dat.csv contains FIA tree-level measurements for 200 sites
#	diameter at breast height in inches
#	tree height in feet at 200 sites as well
#	year they were measured 
#	forest stand age. 
#
#	site_level_lob_covariates.csv contains covariate information for each site 
#	Climate variables: WorldClim: "mat", "t_seas", "t_range", "t_wet_q", "t_warm_q",
#	"map”, "p_wet_q" "p_warm_q" are climate variables from 
#	(https://www.worldclim.org/ ) 
#	Soil variables: Soil Grid 2.0: "sand", "silt", "clay", "bdod", "cec", "cfvo", 
#	"nitrogen","phh2o", and "bdticm" (https://soilgrids.org/)
#=============================================================================
# load libraries
#=============================================================================
library(dplyr)
library(tidyverse)

#Machine learning
library(keras)
library(randomForest)

#Stats with LMEs and GAMMs
library(MuMIn)
library(mgcv) #will also load lme4
library(gamm4)


#=============================================================================
# load data sets and do some initial sanity checks. 
#=============================================================================
trees = read.csv(file="./tree_level_lob_dat.csv")
sites = read.csv(file="./site_level_lob_covariates.csv")
#Match the case in col names
sites = rename(sites, LAT=lat, LON=lon)
set.seed(123)

#=============================================================================
#Clean and merge data sets so trees have all of the site-level data
#=============================================================================
#=============================================================================
#The goal of this section now is to produce three data sets for 
#a series of RF models: 
# 1. A standardized dataset without DIA, PLT_CN, LAT, LON, MEASYEARS, or PLOT to predict
#	 HT.
# 2. A standardized dataset without HT, PLT_CN, LAT, LON, MEASYEARS, or PLOT to predict
#	 DIA. 
# 3. The base dataset for PLT_CN at STDAGE = 25. 
#=============================================================================
#Assign the unique plot IDs
plot_ids = unique(trees[c("LAT", "LON", "PLT_CN")])
plot_ids$NUM = seq_len(nrow(plot_ids))
plot_ids$PLOT= paste0("plot", seq_len(nrow(plot_ids)))

#Add this info as a new column in both trees and sites
trees = merge(trees, plot_ids, by = c("LAT", "LON", "PLT_CN"))
sites = merge(sites, plot_ids, by = c("LAT", "LON", "PLT_CN"))

#Merge so that trees now have the site data. 
tree_sites = left_join(trees, sites, by = c("PLOT","NUM","LAT", "LON", "PLT_CN"))

#Make a separate version of this for the ML fits, where features have all been 
#scaled (mean 0 and SD 1), and categorical variables have been one-hot encoded.
#Remove LAT, LON, PLOT, MEASYEAR for now. 
tree_sites_ml = tree_sites[, !colnames(tree_sites) %in% 
								c("LAT", "LON", "PLOT","NUM", "MEASYEAR"  )]

#Scale
tree_sites_ml = as.data.frame(scale(tree_sites_ml))

#One-hot encoding for PLOTS: 
#Get the variable's unique integer alphabet: 
plts = (plot_ids$NUM)
nplts = length(plts)
mplts = min(plts)

#Use this function from Keras to make the encoding for plots
plts_encoded = to_categorical(tree_sites$NUM-mplts, num_classes = nplts)
colnames(plts_encoded) = plot_ids$PLOT

#Make the three different versions for models
#For HT 
tree_sites_ml_HT = cbind(tree_sites_ml,plts_encoded )
tree_sites_ml_HT = tree_sites_ml_HT[, !colnames(tree_sites_ml_HT) %in% 
								c("DIA", "PLT_CN" )]


#For DIA 
tree_sites_ml_DIA = cbind(tree_sites_ml,plts_encoded )
tree_sites_ml_DIA = tree_sites_ml_DIA[, !colnames(tree_sites_ml_DIA) %in% 
								c("HT", "PLT_CN" )]

#For PLT_CN
tree_sites_ml_CN= cbind(tree_sites_ml,plts_encoded )

#Split data for training and testing: 
ind = sample(2, nrow(tree_sites_ml), replace = TRUE, prob = c(0.9, 0.1))
tree_sites_ml

train_rf_ml_HT  = tree_sites_ml_HT  [ind==1,]
test_rf_ml_HT = tree_sites_ml_HT  [ind==2,]

train_rf_ml_DIA  = tree_sites_ml_DIA  [ind==1,]
test_rf_ml_DIA = tree_sites_ml_DIA  [ind==2,]

train_rf_ml_CN  = tree_sites_ml_CN  [ind==1,]
test_rf_ml_CN = tree_sites_ml_CN  [ind==2,]

#=============================================================================
#Fit RF models for HT, DIA, and then the full model
#=============================================================================
###HT
#Tuning the RF model: 
t = tuneRF(train_rf_ml_HT[,-1], train_rf_ml_HT[,1],
   stepFactor = 0.5,
   plot = TRUE,
   ntreeTry = 150,
   trace = TRUE,
   improve = 0.05)

#Get mtry with the lowest OOB Error
# t[ as.numeric(t[,2]) < 0 ] = 1
mtry_use = as.numeric(t[which(t == min(t),arr.ind=T)[1],1])  

#Basic RF fitting
model_form = "HT ~."
biomass_rf_ml_HT = randomForest (as.formula(model_form),
	data=train_rf_ml_HT, proximity=TRUE, mtry = mtry_use)

#Prediction
pred_test_rf_ml_HT = predict(biomass_rf_ml_HT, test_rf_ml_HT)

#RMSE between predictions and actual
rmse_rf_ml_HT = sqrt( mean((pred_test_rf_ml_HT - test_rf_ml_HT[,1])^2,na.rm=T) )

###DIA
#Tuning the RF model: 
t = tuneRF(train_rf_ml_DIA[,-1], train_rf_ml_DIA[,1],
   stepFactor = 0.5,
   plot = TRUE,
   ntreeTry = 150,
   trace = TRUE,
   improve = 0.05)

#Get mtry with the lowest OOB Error
# t[ as.numeric(t[,2]) < 0 ] = 1
mtry_use = as.numeric(t[which(t == min(t),arr.ind=T)[1],1])  

#Basic RF fitting
model_form = "DIA ~."
biomass_rf_ml_DIA = randomForest (as.formula(model_form),
	data=train_rf_ml_DIA, proximity=TRUE, mtry = mtry_use)

#Prediction
pred_test_rf_ml_DIA = predict(biomass_rf_ml_DIA, test_rf_ml_DIA)

#RMSE between predictions and actual
rmse_rf_ml_DIA= sqrt( mean((pred_test_rf_ml_DIA - test_rf_ml_DIA[,1])^2,na.rm=T) )

###Full
###DIA
#Tuning the RF model: 
t = tuneRF(train_rf_ml_CN[,-1], train_rf_ml_CN[,1],
   stepFactor = 0.5,
   plot = TRUE,
   ntreeTry = 150,
   trace = TRUE,
   improve = 0.05)

#Get mtry with the lowest OOB Error
# t[ as.numeric(t[,2]) < 0 ] = 1
mtry_use = as.numeric(t[which(t == min(t),arr.ind=T)[1],1])  

#Basic RF fitting
model_form = "PLT_CN ~."
biomass_rf_ml_CN = randomForest (as.formula(model_form),
	data=train_rf_ml_CN, proximity=TRUE, mtry = mtry_use)

#Prediction
pred_test_rf_ml_CN = predict(biomass_rf_ml_CN, test_rf_ml_CN)

#RMSE between predictions and actual
rmse_rf_ml_CN= sqrt( mean((pred_test_rf_ml_CN - test_rf_ml_CN[,1])^2,na.rm=T) )

#=============================================================================
#Create the new data set for fitting, where pH has decreased to the minimum
#pH in the data set. 
#=============================================================================
#Replace the phh2o with minimum:  
minph = min(tree_sites_ml$phh2o)
new_data_ml_HT = tree_sites_ml_HT
new_data_ml_HT$phh2o = rep(minph, nrow(tree_sites_ml_HT))

new_data_ml_DIA = tree_sites_ml_DIA
new_data_ml_DIA$phh2o = rep(minph, nrow(tree_sites_ml_DIA))

new_data_ml_CN = tree_sites_ml_CN
new_data_ml_CN$phh2o = rep(minph, nrow(tree_sites_ml_CN))

#Predict HT and DIA for the new dataset
new_data_ml_HT$pred = predict(biomass_rf_ml_HT, newdata = new_data_ml_HT)
new_data_ml_DIA$pred = predict(biomass_rf_ml_DIA, newdata = new_data_ml_DIA)

#Predict PLT_CN using the predicted HT and DIA
new_data_ml_CN$HT = new_data_ml_HT$pred
new_data_ml_CN$DIA = new_data_ml_DIA$pred
new_data_ml_CN$pred = predict(biomass_rf_ml_CN, newdata = new_data_ml_CN)

#=============================================================================
#Plots
#=============================================================================
###Figure 1: Model predictions vs. original CN
#How much biomass is gained or lost? 
bm_loss= (new_data_ml_CN$pred*bsd1 + bmn1)- (new_data_ml_CN$PLT_CN*bsd1 + bmn1)
#Add this to the data frame
new_data_ml_CN$bm_loss = bm_loss
#Order this to look at the most productive locations
cn_ordered=new_data_ml_CN[,c(1:22, 222:223)][order(new_data_ml_CN$bm_loss,decreasing=T),]  
#Add in the plot Numbers, this will get us back to site recommendations
cn_ordered$NUM = tree_sites$NUM[order(new_data_ml_CN$bm_loss,decreasing=T)] 
#E.g., location recommendations based on the top 200: 
unique(head(cn_ordered,200)$NUM)

fig.name = paste("minph_biomass",".pdf",sep="")
pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)

plot(tree_sites$NUM,bm_loss, 
         ylab = "Biomass/CN Difference", 
         xlab = "Plot number", cex.lab =1.3,pch=0)

dev.off()

#What variables drive the gain in the top 100? 
cor_matrix = cor(cn_ordered$bm_loss[1:100], cn_ordered[1:100,1:21])
heatmap(cor_matrix, symm = TRUE, 
        main = "Correlation Matrix of Numeric Variables")


