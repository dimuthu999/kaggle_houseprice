rm(list=ls())
require(FactoMineR)
require(dplyr)
require(dummies)
library(tidyr)
require(pander)
require(randomForest)
library(ggplot2)
library(neuralnet)
require(e1071)
require(rpart)
setwd("E:/Kaggle/House Prices")


no_of_pcoms <- 25

get_factor_variables <- function(col_names,class){
  groups <- as.data.frame(cbind(col_names,class))
  names(groups)<-c("col_name","class")
  rownames(groups)<-NULL
  return(as.vector(groups[groups$class=="factor",]$col_name))
}
as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}
RMSLE <- function(a, p) {
  s <- 0
  n <- 0
  x <- !is.na(a)
  n <- n + sum(x)
  s <- s + sum((log1p(p[x]) - log1p(a[x]))^2)
  return (sqrt(s/n))
}
load_data <- function(frac=0.7) {
  
  for(fname in c("train","test")) {
    data <- read.csv(paste(fname,".csv",sep = ""))
    
    for(col_name in names(data))  {
      cat(col_name,"\n")
      eval(parse(text=paste("data$",col_name,"[is.na(data$",col_name,")]<-median(as.numeric(data$",col_name,"), na.rm = TRUE)",sep="")))
    }
    
    toFactor <- c("MSSubClass",'OverallQual','OverallCond')
    for(f in toFactor)  {
      eval(parse(text=paste("data$",f,"<-as.factor(data$",f,")",sep="")))
    }
    
    if(fname=="train")  {
      org_train_data <<- data
    }
    
    factor_variables <- get_factor_variables(names(data),sapply(data,class))
    data <- dummy.data.frame(data, names = factor_variables)
    
    names(data) <- gsub('([[:punct:]])|\\s+','_',names(data))
    
    eval(parse(text=paste(fname,"_data<-data",sep="")))
  }
  
  drop_test <- names(test_data)[!names(test_data) %in% names(train_data)]
  drop_train <- names(train_data)[!names(train_data) %in% names(test_data)]
  
  drop_train <- drop_train[!drop_train %in% "SalePrice"]
  train_data <- train_data[,!names(train_data) %in% drop_train]
  test_data <- test_data[,!names(test_data) %in% drop_test]
  
  basic_train_data <<-train_data
  basic_test_data <<- test_data
  
  
  prin_comp <- prcomp(train_data[,!colnames(train_data) %in% c("Id","SalePrice")], scale. = T)
  plot(prin_comp$sdev) #=> 20 PCs are good
  pc_weights <- as.data.frame(prin_comp$rotation[,1:no_of_pcoms])

  
  train_data <- cbind(train_data[,colnames(train_data) %in% c("Id","SalePrice")],as.data.frame(as.matrix(train_data[,!colnames(train_data) %in% c("Id","SalePrice")]) %*% as.matrix(pc_weights)))
  
  X_vars <- names(train_data)[3:(no_of_pcoms+2)]
  
  set.seed(20)
  train_data_cluster <- kmeans(train_data[, X_vars], 5, nstart = 20)
  train_data <- cbind(train_data,as.vector(train_data_cluster$cluster))
  names(train_data)[length(names(train_data))]<- "cluster"
  

  
  all_train_data <- train_data
  all_train_data <- cbind(all_train_data[,! colnames(all_train_data) %in% (X_vars)],as.data.frame(scale(all_train_data[,X_vars])))
  train_data <- sample_frac(all_train_data,frac)
  sid <- unique(train_data$Id)

  train_data <<-train_data
  cv_data <<- all_train_data[! (all_train_data$Id %in% sid),]
  test_data <- cbind(test_data[,colnames(test_data) %in% c("Id","SalePrice")],as.data.frame(as.matrix(test_data[,!colnames(test_data) %in% c("Id","SalePrice")]) %*% as.matrix(pc_weights)))
  test_data <- cbind(test_data[,! colnames(test_data) %in% (X_vars)],as.data.frame(scale(test_data[,X_vars])))
  names(test_data)[1]<-"Id"
  test_data <<- test_data
  X_vars <<- X_vars
}

load_data2 <- function(frac=0.7){
  
  for(fname in c("train","test")) {
    data <- read.csv(paste(fname,".csv",sep = ""))
    
    for(col_name in names(data))  {
      cat(col_name,"\n")
      eval(parse(text=paste("data$",col_name,"[is.na(data$",col_name,")]<-median(as.numeric(data$",col_name,"), na.rm = TRUE)",sep="")))
    }
    
    toFactor <- c("MSSubClass",'OverallQual','OverallCond')
    for(f in toFactor)  {
      eval(parse(text=paste("data$",f,"<-as.factor(data$",f,")",sep="")))
    }
    
    if(fname=="train")  {
      org_train_data <<- data
    }
    
    factor_variables <- get_factor_variables(names(data),sapply(data,class))
    data <- dummy.data.frame(data, names = factor_variables)
    
    names(data) <- gsub('([[:punct:]])|\\s+','_',names(data))
    
    eval(parse(text=paste(fname,"_data<-data",sep="")))
  }
  
  drop_test <- names(test_data)[!names(test_data) %in% names(train_data)]
  drop_train <- names(train_data)[!names(train_data) %in% names(test_data)]
  
  drop_train <- drop_train[!drop_train %in% "SalePrice"]
  train_data <- train_data[,!names(train_data) %in% drop_train]
  test_data <- test_data[,!names(test_data) %in% drop_test]
  
  
  basic_train_data <<- sample_frac(train_data,frac)
  sid <- unique(basic_train_data$Id)
  basic_cv_data <<- train_data[! (train_data$Id %in% sid),]
  basic_test_data <<- test_data
  
  prin_comp <- prcomp(train_data[,!colnames(train_data) %in% c("Id","SalePrice")], scale. = T)
  plot(prin_comp$sdev) #=> 20 PCs are good
  pc_weights <- as.data.frame(prin_comp$rotation[,1:no_of_pcoms])
  
  
  train_data <- cbind(train_data[,colnames(train_data) %in% c("Id","SalePrice")],as.data.frame(as.matrix(train_data[,!colnames(train_data) %in% c("Id","SalePrice")]) %*% as.matrix(pc_weights)))
  
  set.seed(20)
  train_data_cluster <- kmeans(train_data[, 3:(no_of_pcoms+2)], 5, nstart = 20)
  train_data <- cbind(train_data,as.vector(train_data_cluster$cluster))
  names(train_data)[length(names(train_data))]<- "cluster"
  
  all_train_data <- train_data
  train_data <- sample_frac(all_train_data,frac)
  sid <- unique(train_data$Id)
  
  X_vars <<- names(train_data)[3:(no_of_pcoms+2)]
  train_data <<-train_data
  cv_data <<- all_train_data[! (all_train_data$Id %in% sid),]
  test_data <- cbind(test_data[,colnames(test_data) %in% c("Id","SalePrice")],as.data.frame(as.matrix(test_data[,!colnames(test_data) %in% c("Id","SalePrice")]) %*% as.matrix(pc_weights)))
  names(test_data)[1]<- "Id"
  test_data<<-test_data
  
 
}

two_stage_regression <- function(model="rf",write=FALSE,resid_model = "rf",no_of_stages=2)  {

  
  imp_vars <- c("GrLivArea","GrLivArea*GrLivArea","YearBuilt*YearBuilt","YearBuilt","FullBath","ExterQualEx","ExterQualFa","ExterQualGd","ExterQualTA",
                "ExterCondEx","ExterCondFa","ExterCondGd","KitchenQualEx","KitchenQualFa","KitchenQualGd","KitchenQualTA",
                "HouseStyle1_5Fin","HouseStyle1_5Unf","HouseStyle1Story","HouseStyle2_5Unf","HouseStyle2Story","HouseStyleSFoyer",
                "HouseStyleSLvl","NeighborhoodBlmngtn","NeighborhoodBlueste","NeighborhoodBrDale","NeighborhoodBrkSide",
                "NeighborhoodClearCr","NeighborhoodCollgCr","NeighborhoodCrawfor","NeighborhoodEdwards","NeighborhoodGilbert",
                "NeighborhoodIDOTRR","NeighborhoodMeadowV","NeighborhoodMitchel","NeighborhoodNAmes","NeighborhoodNoRidge",
                "NeighborhoodNPkVill","NeighborhoodNridgHt","NeighborhoodNWAmes","NeighborhoodOldTown","NeighborhoodSawyer",  
                "NeighborhoodSawyerW","NeighborhoodSomerst","NeighborhoodStoneBr","NeighborhoodSWISU","NeighborhoodTimber",
                "NeighborhoodVeenker","MiscVal","MiscFeatureGar2","MiscFeatureOthr","MiscFeatureShed",
                "MiscFeatureNA","PoolQCEx","PoolQCGd","PoolQCNA","GarageType2Types","GarageTypeAttchd","GarageTypeBasment",
                "GarageTypeBuiltIn","GarageTypeCarPort","GarageTypeDetchd","GarageTypeNA","PoolQCNA*GrLivArea*ExterCondEx"
                ,"PoolQCNA*GrLivArea*YearBuilt","PoolQCNA*GrLivArea*RoofMatlWdShngl","RoofMatlWdShngl")
  
  all_vars <- names(basic_train_data)[2:317]
  
  basic_model_formula <- as.formula(paste("SalePrice~",paste(all_vars,collapse = "+"),sep = ""))
  resid_model_formula <- as.formula(paste("residual~ pred+pred*pred+",paste("pred*",all_vars,collapse = "+",sep = ""),sep = ""))
  #resid_model_formula <- as.formula(paste("residual~ pred+pred*pred",sep = ""))
  
  if (model=="ols") {
    basic_model <- lm(basic_model_formula,basic_train_data)
  } 
  if (model == "rf") {
    basic_model <- randomForest(basic_model_formula,basic_train_data)
  }
  if (model == "svm") {
    basic_model <- svm(basic_model_formula,basic_train_data)
  }
  
  resid_models <- list()
  basic_train_data_resid <- as.data.frame(cbind(basic_train_data,as.data.frame(basic_train_data$SalePrice-predict(basic_model,basic_train_data)),as.data.frame(predict(basic_model,basic_train_data))))
  names(basic_train_data_resid)[length(names(basic_train_data_resid))]<-"pred"
  names(basic_train_data_resid)[length(names(basic_train_data_resid))-1]<-"residual"
  
  
  basic_cv_predict <- as.data.frame(predict(basic_model,basic_cv_data))
  basic_cv_predict <- as.data.frame(cbind(basic_cv_data,basic_cv_data$SalePrice-basic_cv_predict,basic_cv_predict))
  names(basic_cv_predict)[length(names(basic_cv_predict))]<-"pred"
  names(basic_cv_predict)[length(names(basic_cv_predict))-1]<-"residual"
  basic_cv_predict$pred[basic_cv_predict$pred<0] <- quantile(basic_cv_predict$pred,0.05)
  
  for(i in 1:no_of_stages)  {
    cat(i,"\n")
    if (resid_model=="ols") {
      resid_models[[i]] <- lm(resid_model_formula,basic_train_data_resid)
    } 
    if (resid_model == "rf") {
      resid_models[[i]] <- randomForest(resid_model_formula,basic_train_data_resid)
    }
    if (resid_model == "svm") {
      resid_models[[i]] <- svm(resid_model_formula,basic_train_data_resid)
    }
    
    basic_train_data_resid$residual <- predict(resid_models[[i]],basic_train_data_resid)
    basic_train_data_resid$pred <- basic_train_data_resid$pred + basic_train_data_resid$residual
    
    basic_cv_predict$residual <- predict(resid_models[[i]],basic_cv_predict)
    basic_cv_predict$pred <- basic_cv_predict$pred + basic_cv_predict$residual
  }
  
  basic_cv_resid_predict <- as.data.frame(cbind(basic_cv_data$Id,basic_cv_predict$pred,basic_cv_data$SalePrice))
  names(basic_cv_resid_predict) <- c("Id","P","A")
  plot(basic_cv_resid_predict$P,basic_cv_resid_predict$A)
  cat("RMSLE CV: ",RMSLE(basic_cv_resid_predict$A,basic_cv_resid_predict$P),"\n")


  if(write==TRUE) {
    basic_test_predict <- as.data.frame(predict(basic_model,basic_test_data))
    basic_test_data['pred'] <- basic_test_predict$`predict(basic_model, basic_test_data)`
    basic_test_data$pred[basic_test_data$pred<0] <- quantile(basic_test_data$pred,0.05)
    
    for(i in 1:no_of_stages)  {
        
        basic_test_data['pred'] <- basic_test_data$pred+predict(resid_models[[i]],basic_test_data)
        
        
    }
    basic_test_resid_predict <- basic_test_data[,c("Id","pred")]
    names(basic_test_resid_predict) <- c("Id","SalePrice")
    write.csv(basic_test_resid_predict,file="two_stage_2.csv",quote = FALSE,row.names = FALSE)
  }
}
cluster_and_regression <-function(model="ols",clustered=TRUE,cluster_model="rf"){
  # cluster prediction
  if(clustered==TRUE) {
    cluster_train_data <- train_data[train_data$cluster %in% clusters,]
    cluster_rf_fromula <- as.formula(paste('as.factor(cluster) ~ ' ,paste(X_vars,collapse = "+")))
    
    if(cluster_model=="rf") {
      cluster_rf <- randomForest(cluster_rf_fromula,cluster_train_data)
    }
    if(cluster_model=="svm") {
      cluster_rf <- svm(cluster_rf_fromula,cluster_train_data)
    }
    cluster_rf_prediction <- predict(cluster_rf,cv_data[,X_vars])
    cv_data["predicted_cluster"] <- cluster_rf_prediction
    cv_data$predicted_cluster <- as.integer(as.character(cv_data$predicted_cluster))
    confusion <- as.data.frame(table(cv_data$cluster,cv_data$predicted_cluster))
    confusion$Var1 <- as.numeric.factor(confusion$Var1)
    confusion$Var2 <- as.numeric.factor(confusion$Var2)
    acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
    cat("Clustering Prediction Accuracy : ",acc,"\n")
  }
    
  # model fit
  models<-list()
  cv_predict <- NULL
  models_formula <- as.formula(paste("SalePrice~",paste(X_vars,collapse = "+"),sep = ""))
  # ols for each cluster
  if(model=="ols")  {
    cat("Running OLS","\n")
    if(clustered==TRUE) {
      for(cluster in clusters)  {
        models[[cluster]] <- (lm(models_formula,data=train_data[train_data$cluster==cluster,]))
        cat("Cluster ",cluster,"Adj. R sq  -",summary(models[[cluster]])$adj.r.squared,"\n")
      }
    } else {
      models[['all']] <- (lm(models_formula,data=train_data))
      cat("Combined Regression Adj. R sq ",summary(models[['all']])$adj.r.squared,"\n")
    }
  }
  
  if(model=="rf")  {
     cat("Running Random Forests","\n")
    if(clustered==TRUE) {
       for(cluster in clusters)  {
        models[[cluster]] <- (randomForest(models_formula,data=train_data[train_data$cluster==cluster,]))
       }
    } else {
      models[['all']] <- (randomForest(models_formula,data=train_data))
    }
  }
  
  if(model=="nnet") {
    cat("Running Neural Networks","\n")
    if(clustered==TRUE) {
      for(cluster in clusters)  {
        models[[cluster]] <- neuralnet(models_formula,train_data[train_data$cluster==cluster,], hidden = c(5,5),linear.output = TRUE,rep = 2,stepmax = 1e6)
      }
    } else {
      models[['all']] <- neuralnet(models_formula,train_data, hidden = c(20,20),linear.output = TRUE,rep = 5,stepmax = 1e6)#,err.fct="sse",algorithm = 'sag',learningrate = 0.1)
    }
  }
  
  if(model=="svm")  {
    cat("Running SVM","\n")
    if(clustered==TRUE) {
      for(cluster in clusters)  {
        models[[cluster]] <- (svm(models_formula,data=train_data[train_data$cluster==cluster,]))
      }
    } else {
      models[['all']] <- (svm(models_formula,data=train_data))
    }
  }

  # cross validation
  cat("Cross Validation","\n")
  if(model=="nnet") {
    if(clustered==TRUE) {
      cat("Clusterd","\n")
      for(cluster in clusters)  {
        temp <- as.data.frame(predict(models[[cluster]],cv_data[cv_data$predicted_cluster==cluster,]))
        temp <- as.data.frame(cbind(cv_data[cv_data$predicted_cluster==cluster,]$Id,cv_data[cv_data$predicted_cluster==cluster,]$SalePrice,temp))
        names(temp) <- c("Id","A","P")
        cv_predict <- rbind(cv_predict,temp)
      }
    } else {
      cat("Non-clustered","\n")
      temp <- as.data.frame(compute(models[['all']],cv_data[,X_vars]))
      temp <- as.data.frame(cbind(cv_data$Id,cv_data$SalePrice,temp))
      names(temp) <- c("Id","A","P")
      cv_predict <- rbind(cv_predict,temp)
    }
  }

  if(model %in% c("ols","rf","svm"))  {
   if(clustered==TRUE) {
      cat("Clusterd","\n")
      for(cluster in clusters)  {
        temp <- as.data.frame(predict(models[[cluster]],cv_data[cv_data$predicted_cluster==cluster,]))
        temp <- as.data.frame(cbind(cv_data[cv_data$predicted_cluster==cluster,]$Id,cv_data[cv_data$predicted_cluster==cluster,]$SalePrice,temp))
        names(temp) <- c("Id","A","P")
        cv_predict <- rbind(cv_predict,temp)
      }
    } else {
      cat("Non-clustered","\n")
      temp <- as.data.frame(predict(models[['all']],cv_data))
      temp <- as.data.frame(cbind(cv_data$Id,cv_data$SalePrice,temp))
      names(temp) <- c("Id","A","P")
      cv_predict <- rbind(cv_predict,temp)
    }
    cv_predict$P[cv_predict$P<0] <- quantile(cv_predict$P,0.01)
    plot(cv_predict$A,cv_predict$P)
    cat("RMSLE CV: ",RMSLE(cv_predict$A,cv_predict$P),"\n")
  }
  
  
  # submission data set
  cat("Creating Submission Set","\n")
  test_predict <- NULL
  

  if(model %in% c("ols","rf","svm"))  {
    if(clustered==TRUE) {
      cat("Clusterd","\n")
      cluster_rf_prediction <- predict(cluster_rf,test_data[,X_vars])
      test_data["predicted_cluster"] <- cluster_rf_prediction
      test_data$predicted_cluster <- as.integer(as.character(test_data$predicted_cluster))
      
      for(cluster in clusters)  {
        temp <- as.data.frame(predict(models[[cluster]],test_data[test_data$predicted_cluster==cluster,]))
        temp <- as.data.frame(cbind(test_data[test_data$predicted_cluster==cluster,]$Id,temp))
        test_predict <- rbind(test_predict,temp)
      }
    } else {
      cat("Non-clustered","\n")
      temp <- as.data.frame(predict(models[['all']],test_data[,X_vars]))
      temp <- as.data.frame(cbind(test_data$Id,temp))
      test_predict <- rbind(test_predict,temp)
    }
  }
  names(test_predict) <- c("Id","SalePrice")
  write.csv(test_predict,file="test_predict.csv",quote = FALSE,row.names = FALSE)
}

load_data2()
table(train_data$cluster)
clusters <- c(1,4,5) # major clusters
cluster_and_regression(clustered = FALSE) # no_of_pcoms = 175; RMSLE .18
two_stage_regression(no_of_stages = 2,write=TRUE) # RMSLE .156


t1 <- read.csv("two_stage.csv")
t2 <- read.csv("benchmark.csv")
cor(t1$SalePrice,t2$SalePrice)





# DESCRIPTIVES ------------------------------------------------------------


# org_train_data <- cbind(org_train_data,as.vector(train_data_cluster$cluster))
# names(org_train_data)[length(names(org_train_data))]<- "cluster_kmean"

cluster_summary <- org_train_data %>%
  select(SalePrice,BedroomAbvGr,LotArea,LotFrontage) %>%  # Remove the subject column
  mutate(group = org_train_data$cluster_kmean) %>%
  group_by(group) %>%
  summarise_each(funs(mean(., na.rm = TRUE),sd(., na.rm = TRUE))) %>%  # Calculate summary statistics for each group
  gather(variable, value, -group) %>%  # Convert to long
  separate(variable, c("variable", "statistic")) %>%  # Split variable column
  spread(statistic, value) %>%  # Make the statistics be actual columns
  select(group, variable, mean, sd)
pandoc.table(cluster_summary)
