rm(list=ls())
require(FactoMineR)
require(dplyr)
require(dummies)
library(tidyr)
require(pander)
require(randomForest)
setwd("E:/Kaggle/House Prices")

library(dplyr) # for data cleaning
library(ISLR) # for college dataset
library(cluster) # for gower similarity and pam
library(Rtsne) # for t-SNE plot
library(ggplot2) # for visualization

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

load_data <- function() {
  
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
  
  
  prin_comp <- prcomp(train_data[,!colnames(train_data) %in% c("Id","SalePrice")], scale. = T)
  plot(prin_comp$sdev) #=> 20 PCs are good
  pc_weights <- as.data.frame(prin_comp$rotation[,1:no_of_pcoms])

  
  train_data <- cbind(train_data[,colnames(train_data) %in% c("Id","SalePrice")],as.data.frame(as.matrix(train_data[,!colnames(train_data) %in% c("Id","SalePrice")]) %*% as.matrix(pc_weights)))
  
  set.seed(20)
  train_data_cluster <- kmeans(train_data[, 3:(no_of_pcoms+2)], 5, nstart = 20)
  train_data <- cbind(train_data,as.vector(train_data_cluster$cluster))
  names(train_data)[length(names(train_data))]<- "cluster"
  
  all_train_data <- train_data
  train_data <- sample_frac(all_train_data,0.7)
  sid <- unique(train_data$Id)

  train_data <<-train_data
  cv_data <<- all_train_data[! (all_train_data$Id %in% sid),]
  test_data <<- cbind(test_data[,colnames(test_data) %in% c("Id","SalePrice")],as.data.frame(as.matrix(test_data[,!colnames(test_data) %in% c("Id","SalePrice")]) %*% as.matrix(pc_weights)))
}


load_data()
table(train_data$cluster)
clusters <- c(1,4,5) # major clusters

# cluster prediction
cluster_train_data <- train_data[train_data$cluster %in% clusters,]
cluster_rf_fromula <- as.formula(paste('as.factor(cluster) ~ ' ,paste(names(train_data)[3:(no_of_pcoms+2)],collapse = "+")))
cluster_rf <- randomForest(cluster_rf_fromula,cluster_train_data)
cluster_rf_prediction <- predict(cluster_rf,cv_data[,names(cluster_train_data)[3:(no_of_pcoms+2)]])
cv_data["predicted_cluster"] <- cluster_rf_prediction
cv_data$predicted_cluster <- as.integer(as.character(cv_data$predicted_cluster))
confusion <- as.data.frame(table(cv_data$cluster,cv_data$predicted_cluster))
confusion$Var1 <- as.numeric.factor(confusion$Var1)
confusion$Var2 <- as.numeric.factor(confusion$Var2)
acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
cat("Clustering Prediction Accuracy : ",acc,"\n")

# ols for each cluster
ols<-list()
for(cluster in clusters)  {
  lm_formula <- as.formula(paste("SalePrice~",paste(names(train_data)[3:(no_of_pcoms+2)],collapse = "+"),sep = ""))
  ols[[cluster]] <- (lm(lm_formula,data=train_data[train_data$cluster==cluster,]))
  summary(ols[[cluster]])
  cat("Cluster ",cluster,"Adj. R sq  -",summary(ols[[cluster]])$adj.r.squared,"\n")
}

lm_formula <- as.formula(paste("SalePrice~",paste(c(names(train_data)[3:(no_of_pcoms+2)],'cluster'),collapse = "+"),sep = ""))
ols[['all']] <- (lm(lm_formula,data=train_data))
cat("Combined Regression Adj. R sq ",summary(ols[['all']])$adj.r.squared,"\n")


# random forest for each cluster
rf<-list()
for(cluster in clusters)  {
  lm_formula <- as.formula(paste("SalePrice~",paste(names(train_data)[3:(no_of_pcoms+2)],collapse = "+"),sep = ""))
  rf[[cluster]] <- (randomForest(lm_formula,data=train_data[train_data$cluster==cluster,],ntree=5000))
}

# cross validation sample prediction
cv_predict <- NULL
for(cluster in clusters)  {
  temp <- as.data.frame(predict(ols[[cluster]],cv_data[cv_data$predicted_cluster==cluster,]))
  temp <- as.data.frame(cbind(cv_data[cv_data$predicted_cluster==cluster,]$Id,cv_data[cv_data$predicted_cluster==cluster,]$SalePrice,temp))
  names(temp) <- c("Id","A","P")
  cv_predict <- rbind(cv_predict,temp)
}
cv_predict$P[cv_predict$P<0] <- quantile(cv_predict$P,0.01)
plot(cv_predict$A,cv_predict$P)
cat("RMSLE CV: ",RMSLE(cv_predict$A,cv_predict$P))


# test sample prediction
test_predict <- NULL

cluster_rf_prediction <- predict(cluster_rf,test_data[,names(train_data)[3:(no_of_pcoms+2)]])
test_data["predicted_cluster"] <- cluster_rf_prediction
test_data$predicted_cluster <- as.integer(as.character(test_data$predicted_cluster))

for(cluster in clusters)  {
  temp <- as.data.frame(predict(ols[[cluster]],test_data[test_data$predicted_cluster==cluster,]))
  temp <- as.data.frame(cbind(test_data[test_data$predicted_cluster==cluster,]$`test_data[, colnames(test_data) %in% c("Id", "SalePrice")]`,temp))
  test_predict <- rbind(test_predict,temp)
}
names(test_predict) <- c("Id","SalePrice")



write.csv(test_predict,file="test_predict.csv",quote = FALSE,row.names = FALSE)





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

# Cluster Trial -----------------------------------------------------------
# FROM : https://www.r-bloggers.com/clustering-mixed-data-types-in-r/

set.seed(1680) # for reproducibility

library(dplyr) # for data cleaning
library(ISLR) # for college dataset
library(cluster) # for gower similarity and pam
library(Rtsne) # for t-SNE plot
library(ggplot2) # for visualization

college_clean <- College %>%
  mutate(name = row.names(.),
         accept_rate = Accept/Apps,
         isElite = cut(Top10perc,
                       breaks = c(0, 50, 100),
                       labels = c("Not Elite", "Elite"),
                       include.lowest = TRUE)) %>%
  mutate(isElite = factor(isElite)) %>%
  select(name, accept_rate, Outstate, Enroll,
         Grad.Rate, Private, isElite)


gower_dist <- daisy(college_clean[, -1],
                    metric = "gower",
                    type = list(logratio = 3))

summary(gower_dist)
gower_mat <- as.matrix(gower_dist)

college_clean[
  which(gower_mat == min(gower_mat[gower_mat != min(gower_mat)]),
        arr.ind = TRUE)[1, ], ]



sil_width <- c(NA)

for(i in 2:10){
  
  pam_fit <- pam(gower_dist,
                 diss = TRUE,
                 k = i)
  
  sil_width[i] <- pam_fit$silinfo$avg.width
  
}

# Plot sihouette width (higher is better)

plot(1:10, sil_width,
     xlab = "Number of clusters",
     ylab = "Silhouette Width")
lines(1:10, sil_width)

pam_fit <- pam(gower_dist, diss = TRUE, k = 3)

pam_results <- college_clean %>%
  dplyr::select(-name) %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))

pam_results$the_summary


tsne_obj <- Rtsne(gower_dist, is_distance = TRUE)

tsne_data <- tsne_obj$Y %>%
  data.frame() %>%
  setNames(c("X", "Y")) %>%
  mutate(cluster = factor(pam_fit$clustering),
         name = college_clean$name)

ggplot(aes(x = X, y = Y), data = tsne_data) +
  geom_point(aes(color = cluster))


#######################

gower_dist <- daisy(train_data[, !names(train_data) %in% c('SalesPrice','Id')],metric = "gower",type = list(logratio = 3))
gower_mat <- as.matrix(gower_dist)

sil_width <- c(NA)
for(i in 2:10){
  pam_fit <- pam(gower_dist,diss = TRUE, k = i)
  sil_width[i] <- pam_fit$silinfo$avg.width
}
plot(1:10, sil_width, xlab = "Number of clusters", ylab = "Silhouette Width")
lines(1:10, sil_width)

pam_fit <- pam(gower_dist, diss = TRUE, k = 5)
pam_results <- train_data %>%
  dplyr::select(-Id) %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))
pam_results <- train_data %>%
  dplyr::select(SalePrice) %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.),the_mean = mean(.,na.rm=TRUE))
pam_results$the_summary


################### missal




res <- MFA(train_data[,!colnames(train_data) %in% c("Id","SalePrice")], group=c(17,4,15,4,8,19,8,2,2), type=c("n","n","s","n","s","n","s","s","n"),ncp=5)








data(wine)
res <- MFA(wine, group=c(2,5,3,10,9,2), type=c("n",rep("s",5)),
           ncp=10, name.group=c("orig","olf","vis","olfag","gust","ens"),
           num.group.sup=c(1,6))

res$global.pca$
  
  data(iris)
log.ir <- log(iris[, 1:4])
ir.species <- iris[, 5]

# apply PCA - scale. = TRUE is highly 
# advisable, but default is FALSE. 
ir.pca <- prcomp(log.ir,
                 center = TRUE,
                 scale. = TRUE) 