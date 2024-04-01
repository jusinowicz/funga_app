#=============================================================================
# Instructions:
#	1. Use the attached datasets to model biomass.
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
#This section is exploratory and can be commented out of final code.
#=============================================================================

head(trees)
head(sites)

#Match the case in col names
sites = rename(sites, LAT=lat, LON=lon) 

#Look for any NAs or strange non-finite values: 
in_trees = sum(is.na(trees))
if_trees = sum(apply(trees,2,is.infinite)) 

#How are sets organized? By site?
#Add a new column "PLOT" to the tree level data
site_ids = unique(trees[c("LAT", "LON")])
site_ids$SITE = paste0("site", seq_len(nrow(site_ids)))

#Number of unique lat/lon combos is <200. So some lat/lon must have 
#multiple plots within them. Look at a table quick: 
data.frame(table(sites$LON)) 
#Looks like some lat/lon have 2 occurrences. That means plots are nested
#within sites. Need a way to identify plots within a site. 
#PLT_CN is the only option. Will this work? Assumes that 2 plots within a
#site have unique measures of CN:

plot_ids = unique(trees[c("LAT", "LON", "PLT_CN")])
pl1 = arrange(plot_ids, LAT,LON,PLT_CN)

#Check these against sites for uniqueness: 
plot_ids2 = unique(sites[c("LAT", "LON", "PLT_CN")])
pl2 = arrange(plot_ids2, LAT,LON,PLT_CN)
identical(pl1,pl2)

#Returns TRUE, so this is ok! 

#Quickly plot biomass (i.e. PLT_CN) vs diameter and biomass vs. height, 
#since we'll be looking for the way that dia and ht predict biomass given
#site features. 

# Scatter plot of DIA vs. PLT_CN
p1 = ggplot(trees)+ 
  geom_point(aes(x = DIA, y = PLT_CN)) +
  labs(x = "DIA", y = "Biomass") # +
  #theme_bw() +

# Scatter plot of HT vs. PLT_CN
p2 = ggplot(trees)+ 
  geom_point(aes(x = HT, y = PLT_CN)) +
  labs(x = "HT", y = "Biomass")# +
  #theme_bw() 

# Scatter plot of STDAGE vs. PLT_CN
p3 = ggplot(trees)+ 
  geom_point(aes(x = STDAGE , y = PLT_CN)) +
  labs(x = "HT", y = "Biomass")# +
  #theme_bw() 
 
ggarrange(p1,p2,p3)

#What is the spatial scale of these plots like? Look quickly at spatial 
#autocorrelation in biomass using variograms to see if this will be a factor.
library(sp)
library(gstat)

trees_sp = trees
coordinates(trees_sp) = ~LAT+LON
v1 = variogram(PLT_CN~1, trees_sp, alpha=c(0,45,90,135))                                                      
plot(v1)    
#Doesn't look like we're at a spatial scale where this matters. 

#=============================================================================
#Clean and merge data sets so trees have all of the site-level data
#=============================================================================
#Now assign the unique plot IDs
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
#For plotting later: 
bmn1 = mean(tree_sites$PLT_CN) 
bsd1 = sqrt(var(tree_sites$PLT_CN))

#One-hot encoding: 
#Get each variable's unique integer alphabet: 
yrs = as.integer(unique (tree_sites$MEASYEAR))
nyrs = length(yrs)
myrs = min(yrs)

plts = (plot_ids$NUM)
nplts = length(plts)
mplts = min(plts)

#Use this function from Keras to make the encoding for years
yrs_encoded = to_categorical(tree_sites$MEASYEAR-myrs, num_classes = nyrs)
yr_names = vector("character",nyrs) #Add names
for(t in 1:nyrs){
    		yr_names[t] = paste("Y",t,sep="")
}
colnames(yrs_encoded) = yr_names 

#Use this function from Keras to make the encoding for plots
plts_encoded = to_categorical(tree_sites$NUM-mplts, num_classes = nplts)
colnames(plts_encoded) = plot_ids$PLOT

#Make three different versions for model exploration from different
#combinations of data frames
#Keep it without any categoricals
tree_sites_ml_nocat = tree_sites_ml
#Just plots
tree_sites_ml_noyrs = cbind(tree_sites_ml,plts_encoded )
#Plots and years
tree_sites_ml = cbind(tree_sites_ml,yrs_encoded,plts_encoded )

#Make a scaled version of tree_sites for lmer
tree_sites2 = tree_sites[, !colnames(tree_sites) %in% 
                c("LAT", "LON", "PLOT","NUM", "MEASYEAR"  )]

#Scale
tree_sites2 = as.data.frame(scale(tree_sites2))
tree_sites2 = cbind(tree_sites[, colnames(tree_sites) %in% 
                c("LAT", "LON", "PLOT","NUM", "MEASYEAR"  )], tree_sites2 )

#Split data for training and testing: 
ind = sample(2, nrow(tree_sites_ml), replace = TRUE, prob = c(0.7, 0.3))
train_rf_nocat = tree_sites_ml_nocat [ind==1,]
test_rf_nocat = tree_sites_ml_nocat [ind==2,]

train_rf_noyrs = tree_sites_ml_noyrs [ind==1,]
test_rf_noyrs = tree_sites_ml_noyrs [ind==2,]

train_rf= tree_sites_ml [ind==1,]
test_rf = tree_sites_ml [ind==2,]

train_stats = tree_sites [ind==1,]
test_stats = tree_sites [ind==2,]

train_stats2 = tree_sites2 [ind==1,]
test_stats2 = tree_sites2 [ind==2,]

#=============================================================================
#First, fit a RandomForest ML model. Use it to start looking at variable 
#importance.  
#=============================================================================
#Tuning the full RF model: 
t = tuneRF(train_rf[,-1], train_rf[,1],
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
biomass_rf = randomForest (as.formula(model_form),
	data=train_rf, proximity=TRUE, mtry = mtry_use)

#Prediction
pred_test_rf = predict(biomass_rf, test_rf)

#RMSE between predictions and actual
rmse_rf = sqrt( mean((pred_test_rf - test_rf[,1])^2,na.rm=T) )

#Look at variable importance: 
#fig.name = paste("varImpPlot3",".pdf",sep="")
#pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)

p1 = varImpPlot(biomass_rf,
           sort = T,
           n.var = 40,
           main = "Variable Importance"
)
    
#One thing that is immediately apparent from this plot is that the year has a 
#massive impact. Will need to control for year in any statistical models, probably
#as a random effect. Try fitting the model without them and see what pops
#next.  

#dev.off()

#####RF model 2: no years
#Tuning the full RF model: 
t = tuneRF(train_rf_noyrs[,-1], train_rf_noyrs[,1],
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
biomass_rf_noyrs = randomForest (as.formula(model_form),
  data=train_rf_noyrs, proximity=TRUE, mtry = mtry_use)

#Prediction
pred_test_rf_noyrs = predict(biomass_rf_noyrs, test_rf_noyrs)

#RMSE between predictions and actual
rmse_rf_noyrs = sqrt( mean((pred_test_rf_noyrs - test_rf_noyrs[,1])^2,na.rm=T) )

#Look at variable importance: 
#fig.name = paste("varImpPlot3",".pdf",sep="")
#pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)

p2 = varImpPlot(biomass_rf_noyrs,
           sort = T,
           n.var = 40,
           main = "Variable Importance"
)
    
#dev.off()
#Ok these results are a bit more intuitive. Nitrogen, plot120 (what is going on with
#this plot???), STDAGE, cfvo, clay,then
#a chunk of climate variables, then the plot effects. 
#Where is pH? 

#####RF model 3: no yrs, no plots
#Tuning the full RF model: 
t = tuneRF(train_rf_nocat[,-1], train_rf_nocat[,1],
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
biomass_rf_nocat = randomForest (as.formula(model_form),
  data=train_rf_nocat, proximity=TRUE, mtry = mtry_use)

#Prediction
pred_test_rf_nocat = predict(biomass_rf_nocat, test_rf)

#RMSE between predictions and actual
rmse_rf_nocat = sqrt( mean((pred_test_rf_nocat - test_rf_nocat[,1])^2,na.rm=T) )

#Look at variable importance: 
#fig.name = paste("varImpPlot3",".pdf",sep="")
#pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)

p3 = varImpPlot(biomass_rf_nocat,
           sort = T,
           n.var = 40,
           main = "Variable Importance"
)
    
#There's phh20, 3 from the bottom. 
#dev.off()

#=============================================================================
#Fit a RNN (just distributed layers for now?). Compare the RMSE with RF model.
#The RSME is an order of magnitude larger than the RF. After comparison, it's 
#clear that this is a less appropriate model structure.
#Let's not waste more time exploring RNNs at the moment.
#=============================================================================
#Model form 
#DNN
build_and_compile_dnn = function() {
  model = keras_model_sequential() %>%
    layer_dense(128, activation = 'relu',input_shape = c(ncol(tree_sites_ml)-1)) %>%
    layer_dense(64, activation = 'relu') %>%
    layer_dense(32, activation = 'relu') %>%
    layer_dense(1)

  model %>% compile(
    loss = 'mean_absolute_error',
    optimizer = optimizer_adam(),
    metrics = c('mse')
  )
	  model
}

##########################################################################
#Create the save points for models. This can help with reproducibility
#as well as future training.
##########################################################################
dnn_checkpoint_path = paste("./DNN/", "biomassDNN",".tf", sep="")
dnn_checkpoint_dir = fs::path_dir(dnn_checkpoint_path)

dnn_log = "logs/run_dnn"

# Create a callback that saves the model's weights
dnn_callback = callback_model_checkpoint(
  filepath = dnn_checkpoint_path,
  #save_weights_only = TRUE,
  verbose = 1
)

#Build the models
biomass_DNN = build_and_compile_dnn()

# Train the model. Use the same training data set from the RF
history_DNN = biomass_DNN %>% fit(
  train_rf[,-1], train_rf[,1],  # Training data
  epochs = 20,       # Number of epochs
  batch_size = 32,   # Batch size
  validation_split = 0.2,  # Validation split
  callbacks = list(dnn_callback, 
				callback_tensorboard(dnn_log )) # Pass callback to training
)

# Plot training history
plot(history_DNN)

#Predict the test data and get RMSE
test_data = array( test_rf[,-1], 
					 dim = c(1, dim(test_rf[,-1])[1], dim(test_rf[,-1])[2] ) )

lf_dnn = biomass_DNN  %>% predict(test_data[1,,]) 
rmse_dnn = sqrt(mean((lf_dnn -test_rf[,1])^2,na.rm=T))

#=============================================================================
#Fit a more classic statistical model: Linear Mixed Effects Model 
#Let's see if we can get some second opinions on variable importance. 
#=============================================================================
model_form_lmer = "PLT_CN ~ 
        STDAGE + phh2o+ 
        DIA+ HT + 
        mat+t_seas + t_range + t_wet_q+t_warm_q+
        p_wet_q+p_warm_q + 
        sand + silt + clay + bdod + cec+cfvo +
        nitrogen+bdticm +(1|MEASYEAR)"

#Fit with lmer
#Remember to set na.action = "na.fail" for dredge to work 
biomass_lmer = lmer(as.formula(model_form_lmer), data = train_stats2, na.action = "na.fail")  

#Predict and get RSME
lf_lmer = predict (biomass_lmer, test_stats2)
rsme_lmer = sqrt(mean((lf_lmer -test_stats2$PLT_CN)^2,na.rm=T))

#See what info we can get about the possible model space and variable
#importance. Ok this is taking too long to run for this assignment. 
#biomass_lmer_dr = dredge(biomass_lmer) 

# For now, look at standardized coefficient estimates
bl_coef = coef(summary(biomass_lmer))
bl_plot = data.frame( var = rownames(bl_coef), coef = abs(bl_coef[, "Estimate"]))
bl_plot = bl_plot[order(bl_plot[,2], decreasing = T ),]  

# Plot variable importance
barplot(abs(bl_coef[, "Estimate"]), names.arg = rownames(bl_coef),
        main = "Variable Importance", xlab = "Predictor Variables", ylab = "Absolute Coefficient Estimate")

#=============================================================================
#Fit a more classic statistical model using a Generalized Additive 
#Mixed-effects Model (GAMM).
#Running this the right way with either mgcv or gamm4 is not working at the 
#moment. Taking too much time for this assignment. Instead, I'm going to do 
#this in a stepwise way that is not technically the best but works to account 
#for some of the variance in the random effects: 
#First fit the random effects only with lmer
#Then fit the GAM to the residuals  
#=============================================================================
remove_re = "PLT_CN ~ (1|MEASYEAR)"

#Fit with lmer
#Remember to set na.action = "na.fail" for dredge to work 
re_resid = lmer(as.formula(remove_re), data = train_stats2, na.action = "na.fail")  
gam_dat_tmp = residuals(re_resid)
gam_dat = train_stats2
gam_dat$PLT_CN = gam_dat_tmp

model_form = "PLT_CN ~ 
        s(STDAGE) + s(phh2o)+ 
        s(DIA)+ s(HT) + 
        s(mat)+s(t_seas)+ s(t_range) + s(t_wet_q)+s(t_warm_q)+
        s(p_wet_q)+s(p_warm_q)+ 
        s(sand) + s(silt) + s(clay)+ s(bdod) + s(cec)+s(cfvo)+
        s(nitrogen)+s(bdticm)"

#Use select=TRUE to do comparative model-fitting in mgcv. This will tell the smoother
#selection algorithm that it can "penalize away" non-significant smooth terms, i.e reduce
#their effective degrees of freedom to zero. 

biomass_gam = gam(as.formula(model_form), select=TRUE,method="REML", data =gam_dat)
#biomass_gamm = gamm4(as.formula(model_form), random = ~(1|MEASYEAR), data = tree_sites)

#Predict and get RSME (remember to add the random effects back in)
lf_gamm = predict (biomass_gam, test_stats2) + predict (re_resid, test_stats2)
rsme_gamm = sqrt(mean((lf_gamm -test_stats2$PLT_CN)^2,na.rm=T))

#For variable contribution ranking in the GAMM, try the following: 
# Get the residual deviance
residual_deviance = deviance(biomass_gam)

# Get the deviance explained by each smooth term
dev_explained = summary(biomass_gam)$s.table[, "edf"] 

# Calculate the total deviance explained by all smooth terms
total_dev_explained = sum(dev_explained)

# Calculate the percentage of variance explained by each smooth term
percent_var_explained = (dev_explained / total_dev_explained) * 100

#=============================================================================
#=============================================================================
#Plotting
#=============================================================================
###Figure 1: Model predictions vs. test data
fig.name = paste("model_predictions",".pdf",sep="")
pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)
plot(test_stats2$NUM, test_stats2$PLT_CN, ylab = "Biomass/CN", 
          xlab = "Plot number", cex.lab =1.3,pch=0)
points(test_stats2$NUM, pred_test_rf, col="red")
points(test_stats2$NUM, lf_lmer, col="blue", pch=1)
points(test_stats2$NUM, lf_gamm, col="green", pch=2)

dev.off()

###Figure 2: Compare ML rankings: 
fig.name = paste("RF_varimpo",".pdf",sep="")
pdf(file=fig.name, height=8, width=8, onefile=TRUE, family='Helvetica', pointsize=16)

par(mfrow=c(1,3))
#Full model
varImpPlot(biomass_rf,
           sort = T,
           n.var = 40,
           main = "Variable Importance, Full Model"
)

varImpPlot(biomass_rf_noyrs,
           sort = T,
           n.var = 40,
           main = "Variable Importance, No Years"
)
    
varImpPlot(biomass_rf_nocat,
           sort = T,
           n.var = 40,
           main = "Variable Importance, No Years/No Plots"
)

dev.off()


#Save the fitted models! 
save(file="full_models.var", biomass_rf, biomass_rf_nocat, 
  biomass_rf_noyrs, biomass_gam, biomass_lmer)




















































































































