#!/usr/bin/env Rscript

# clear environment
rm(list =ls())

# SET WORKING DIRECTORY
setwd("~/Documents/desktop/ML/R/problems/nnetR")



#-------------------------------------------------------------------------------------------------------------------------
#Install and load packages necessary 
#-------------------------------------------------------------------------------------------------------------------------
if (!require("stringdist")) {
  install.packages("stringdist")
  library(stringdist)
}

if (!require("sqldf")) {
  install.packages("sqldf")
  library(sqldf)
}

if (!require("RODBC")) {
  install.packages("RODBC")
  library(RODBC)
}

if (!require("RJSONIO")) {
  install.packages("RJSONIO")
  library(RJSONIO)
}

if (!require("RCurl")) {
  install.packages("RCurl")
  library(RCurl)
}

if (!require("dplyr")) {
  install.packages("dplyr")
  library(dplyr)
}

if (!require("plyr")) {
  install.packages("plyr")
  library(plyr)
}

if (!require("ggplot2")) {
  install.packages("ggplot2")
  library(ggplot2)
}

if (!require("reshape2")) {
  install.packages("reshape2")
  library(reshape2)
}

if (!require("neuralnet")) {
  install.packages("neuralnet")
  library(neuralnet)
}

if (!require("nnet")) {
  install.packages("nnet")
  library(nnet)
}


## Variables
TRAIN_PERCENT = 0.8 ## you can vary between 0.1 to 0.9
HIDDEN_LAYER = 20   ## you can vary between 1 and 50 but will depend on the number of attributes


## check source files in correct location
# ensure data present
if (file.exists("mushroom.csv") !=TRUE) {
  stop(paste("File mushroom.csv not located in ", this.dir, ". Please correct."), call. = FALSE)
}


## Load data
allmushroomraw <- read.csv("mushroom.csv", stringsAsFactors = F, header=T)
allmushroom<-as.data.frame(lapply(dplyr::select(allmushroomraw,-ID), as.factor))


# get the training and test sets
mushroomtrainindx<-sample(1:nrow(allmushroom), round(TRAIN_PERCENT*nrow(allmushroom)), replace=FALSE)
mushroomtestindx<-setdiff(1:nrow(allmushroom), mushroomtrainindx)
mushroomtrain<-allmushroom[mushroomtrainindx,]
mushroomtest<-allmushroom[mushroomtestindx,]


## train the neural network
## You can vary the features used to train the model here
## CapShape,CapSurf,CapColor,Bruises,Odor,GillAtt,GillSpace,GillSize,GillColor,StalkShape,StalkRoot,SsurfAbove,SsurfBelow,ScolorAbove,ScolorBelow,VeilColor,RingNum,RingType,Spore,Pop,Habitat
## mushroomnet<-nnet(EorP~CapShape+CapSurf+CapColor+Bruises+Odor+GillAtt+GillSpace+GillSize+GillColor+StalkShape+StalkRoot+SsurfAbove+SsurfBelow+ScolorAbove+ScolorBelow+VeilColor+RingNum+RingType+Spore+Pop+Habitat, mushroomtrain, size=HIDDEN_LAYER)
#mushroomnet<-nnet(EorP~CapShape+CapSurf+CapColor+Bruises+Odor+GillAtt, mushroomtrain, size=HIDDEN_LAYER, na.action="na.omit")

mushroomnet<-nnet(EorP~CapShape+Bruises, mushroomtrain, size=HIDDEN_LAYER, na.action="na.omit")
## use the test data and generate predictions
guessclass<-predict(mushroomnet, mushroomtest, type = "class")

table(actual=mushroomtest$EorP,predicted=guessclass)

df <- table(actual=mushroomtest$EorP,predicted=guessclass)


## use this code if you want to see more details
results <- data.frame(mushroomtest, prediction = guessclass)
e_e_results<-sqldf("select * from results where results.EorP = 'e' and results.prediction = 'e'")
p_e_results<-sqldf("select * from results where results.EorP = 'p' and results.prediction = 'e'")
p_p_results<-sqldf("select * from results where results.EorP = 'p' and results.prediction = 'p'")
e_p_results<-sqldf("select * from results where results.EorP = 'e' and results.prediction = 'p'")

# plot using ggplot2 based on different expected values
expvshidden.df <- data.frame(Expected.Value=c(6486,2714,-658,-1884, 3356),
                             HIDDEN_LAYER=c(5, 10, 15, 18, 20))


ggplot(expvshidden.df, aes(x=HIDDEN_LAYER, y=Expected.Value)) + 
  geom_point() + 
  geom_line()

expvshidden.df1 <- data.frame(Expected.Value=c(-23470,15526,8610,13406, 6194),
                              HIDDEN_LAYER=c(5, 10, 15, 18, 20))


ggplot(expvshidden.df1, aes(x=HIDDEN_LAYER, y=Expected.Value)) + 
  geom_point() + 
  geom_line()

