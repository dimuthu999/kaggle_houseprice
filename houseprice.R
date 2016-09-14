rm(list=ls())
require(FactoMineR)
require(dplyr)
require(dummies)
setwd("E:/Kaggle/House Prices")

library(dplyr) # for data cleaning
library(ISLR) # for college dataset
library(cluster) # for gower similarity and pam
library(Rtsne) # for t-SNE plot
library(ggplot2) # for visualization


train_data <- read.csv(url("https://github.com/dimuthu999/kaggle_houseprice/raw/master/train.csv"))
toFactor <- c("MSSubClass",'OverallQual','OverallCond')
for(f in toFactor)  {
  eval(parse(text=paste("train_data$",f,"<-as.factor(train_data$",f,")",sep="")))
}


# 1-type; 2-area; 3-size; 4-ammenities; 5-quality; 6-sale time; 7-sale-type
groups <- c(0,1,2,3,3,1,1,1,1,4,1,1,2,2,2,1,1,5,5,5,5,1,1,1,1,1,3,5,5,1,1,5,1,5,3,5,3,3,3,4,5,4,5,3,3,5,3,4,4,4,4,4,4,5,3,5,4,5,4,5,5,4,3,5,5,5,5,5,3,3,3,5,5,5,5,5,6,6,7,7,0)
groups <- as.data.frame(cbind(names(train_data),groups,sapply(train_data,class)))
names(groups)<-c("col_name","group","class")
rownames(groups)<-NULL
groups<-groups[ order(groups$group,groups$class), ]


train_data <- train_data[,as.vector(groups$col_name)]
table(groups$group,groups$class)

factor_variables <- as.vector(groups[groups$class=="factor",]$col_name)
train_data <- dummy.data.frame(train_data, names = factor_variables)

names(train_data) <- gsub('([[:punct:]])|\\s+','_',names(train_data))

for(col_name in names(train_data))  {
  cat(col_name,"\n")
  eval(parse(text=paste("train_data$",col_name,"[is.na(train_data$",col_name,")]<-median(train_data$",col_name,", na.rm = TRUE)",sep="")))
}

prin_comp <- prcomp(train_data[,!colnames(train_data) %in% c("Id","SalePrice")], scale. = T)
plot(prin_comp$sdev) #=> 20 PCs are good
pc_weights <- as.data.frame(prin_comp$rotation[,1:20])

train_data <- cbind(train_data[,colnames(train_data) %in% c("Id","SalePrice")],as.data.frame(as.matrix(train_data[,!colnames(train_data) %in% c("Id","SalePrice")]) %*% as.matrix(pc_weights)))

lm_formula <- as.formula(paste("SalePrice~",paste(names(train_data)[3:22],collapse = "+"),sep = ""))
summary(lm(lm_formula,data=train_data))







glimpse(train)

gower_dist <- daisy(train[, !names(train) %in% c('SalesPrice','Id')],
                    metric = "gower",
                    type = list(logratio = 3))

summary(gower_dist)
gower_mat <- as.matrix(gower_dist)


sil_width <- c(NA)

for(i in 2:10){
  
  pam_fit <- pam(gower_dist,
                 diss = TRUE,
                 k = i)
  
  sil_width[i] <- pam_fit$silinfo$avg.width
  
}


plot(1:10, sil_width,
     xlab = "Number of clusters",
     ylab = "Silhouette Width")
lines(1:10, sil_width)

pam_fit <- pam(gower_dist, diss = TRUE, k = 2)

pam_results <- train %>%
  dplyr::select(-Id) %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))


pam_results <- train %>%
  dplyr::select(SalePrice,BsmtFinSF1,TotalBsmtSF,FullBath,TotRmsAbvGrd) %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.),the_mean = mean(.,na.rm=TRUE))

pam_results$the_summary



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