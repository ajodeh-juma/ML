# Predicting income per person per city using city, region and country data

# The objective of this model is to reduce/minimize Root Mean Square Error (RMSE)

#This model will measure the average income per city, region and country and use them as features to train an xgboost regression model
rm(list=ls())
setwd("~/Desktop/ML/R/problems/predict_income/")
options(scipen=20)

#==================================== load packages =================================#

lapply(c("data.table", "ggplot2", "xgboost", "DiagrammeR", "devtools"), require, character.only=T)

#==================================== helper functions ==============================#

# Root Mean Squared Error (RMSE)

rmse <- function(preds, actuals) sqrt(mean((preds-actuals)^2))

# split a vector into a list of vectors of equal(near equal) sizes

chunk <- function(x, n) split(x, cut(seq_along(x), n, labels=FALSE))

# Load data (current working directory is predict_income problem directory)

train <- fread("data/train.csv")
test <- fread("data/test.csv")
setnames(test, "Income", "IncomeTruth")

#==================================== Build the modified training dataset ====================#

transformTrain <- function(folds=5){
  # splits the training set into disjoint (train,, test) pairs: {(train1, test1), (train2, test2), ....} for
  # number of specified folds.
  # For a given (train_k, test_k) pair, the incomes in the train_k are averaged by city, region and country (separately)
  # and then inserted into test_k appropriately.
  # Finally, the test sets are concatenated, producing a new training dataset
  
  test_folds <- chunk(sample(nrow(train), nrow(train)), folds)
  train_folds <- lapply(test_folds, function(testIdxs) seq(nrow(train))[-testIdxs])
  
  tests <- lapply(seq_len(folds), FUN=function(i){
    train1 <- train[train_folds[[i]]]
    train1_countries <- train1[, list(Countries=.N, CountryAvg=mean(Income)), by=list(CountryID)]
    train1_regions <- train1[, list(Regions=.N, RegionAvg=mean(Income)), by=list(RegionID)]
    train1_cities <- train1[, list(Cities=.N, CityAvg=mean(Income)), by=list(CityID)]
    test1 <- train[test_folds[[i]]]
    test1 <- train1_countries[test1, on="CountryID"]
    test1 <- train1_regions[test1, on="RegionID"]
    test1 <- train1_cities[test1, on="CityID"]
    return(test1)
  })
  
  # Build the new training dataset by concatenating all the test sets
  train_new <- rbindlist(tests, use.names=TRUE)
  
  # Return a list of the trainIdxs, testIdxs, and the new training dataset
  
  return(list(trainIdxs=train_folds, testIdxs=test_folds, train=train_new))
  
}

# create the new training set

transformed <- transformTrain(5)
train_new <- transformed[["train"]]
trainIdxs <- transformed[["trainIdxs"]]
testIdxs <- transformed[["testIdxs"]]

# create the modified test set
countryAvgs <- train[, list(Countries=.N, CountryAvg=mean(Income)), keyby=CountryID]
regionAvgs <- train[, list(Regions=.N, RegionAvg=mean(Income)), keyby=RegionID]
cityAvgs <- train[, list(Regions=.N, CityAvg=mean(Income)), keyby=CityID]
test <- countryAvgs[test, on="CountryID"]
test <- regionAvgs[test, on="RegionID"]
test <- cityAvgs[test, on="CityID"]

#=============================== xgboost ================================#

features <- c("Cities", "CityAvg", "Regions", "RegionAvg", "Countries", "CountryAvg")

# Train model
paramList <- list(eta=.2, gamma=0, max.depth=3, min_child_weight=1, subsample=.9, colsample_bytree=1) # test various hyperparameters and values
bst.cv <- xgb.cv(params=paramList, data=as.matrix(train_new[, features, with=FALSE]), label=as.matrix(train_new$Income), folds=testIdxs, eval_metric="rmse", early_stopping_rounds=3, nrounds=200, prediction=TRUE)
bst <- xgboost(params=paramList, data=as.matrix(train_new[, features, with=FALSE]), label=as.matrix(train_new$Income), nrounds=3)

# Predict & Evaluate

#---------------------------- Predict -------------------------#

train[, IncomeXGB := predict(bst, as.matrix(train_new[, features, with=FALSE]))]
test[, IncomeXGB := predict(bst, as.matrix(test[, features, with=FALSE]))]

# Trees
bst.trees <- xgb.model.dt.tree(features, model=bst)
bst.trees[Tree==0]

#---------------------------- Importance ----------------------#

xgb.importance(model=bst, features)

# plot a minimal model (3 rounds of training)

bst.minimal <- xgboost(params=paramList, data=as.matrix(train_new[, features, with=FALSE]), label=as.matrix(train_new$Income), nrounds=3)
png("tree.png", width = 8000, height = 6000)
xgb.plot.tree(features, model=bst.minimal)
dev.off()

#--------------------------- Evaluate -----------------------#
rmse(train$IncomeXGB, train$Income)
rmse(test$IncomeXGB, test$IncomeTruth)

#Errors
train[, SE := (IncomeXGB-Income)^2]
test[, SE := (IncomeXGB-IncomeTruth)^2]
test[order(SE)]

