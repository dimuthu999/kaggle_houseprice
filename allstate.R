rm(list = ls())
setwd("E:/allstate")
require(sqldf)
require(MASS)
require(dplyr)
require(plyr)
require(nnet)
require(neuralnet)
require(ROCR)
require(randomForest)
require(e1071)
require(rpart)


# DESCRIPTIVE STATS -------------------------------------------------------
rm(list = ls())
setwd("E:/allstate")
as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}
require(dplyr)
require(tidyr)
require(stargazer)
traindata <- read.csv("train.csv")
traindata <- traindata %>% 
  arrange(customer_ID, shopping_pt) %>%
  group_by(customer_ID) %>%
  dplyr::mutate(rank=row_number())
traindata <- as.data.frame(traindata)

purchaseddata <- traindata[traindata$record_type==1,]
purchaseddata['age_difference']<- purchaseddata$age_oldest - purchaseddata$age_youngest

cols <- c('car_age','risk_factor','age_oldest','age_youngest','age_difference','group_size','married_couple','duration_previous','homeowner')

purchaseddata[,cols][,names(Filter(is.factor, purchaseddata[, cols]))] <- (as.character(purchaseddata[,cols][,names(Filter(is.factor, purchaseddata[, cols]))]))

temp <- as.numeric.factor(purchaseddata[,cols][,names(Filter(is.factor, purchaseddata[, cols]))])


stargazer(
  purchaseddata[, cols], type = "html", 
  summary.stat = c("mean", "sd","min", "p25", "median", "p75", "max"),out="decstats.htm"
)

stargazer(cor(purchaseddata[, cols],use="complete.obs"),type="html",out="corr.htm")
stargazer(cor(purchaseddata[, c(cols,"A","B","C","D","E","F","G")],use="complete.obs"),type="html",out="corr2.htm")

cols_q <- c('car_age','age_oldest','age_youngest','duration_previous','cost')
for(col in cols_q) {
  cat(col,"\n")
  eval(parse(text=paste("q <- quantile(purchaseddata$",col,",c(0,0.25,0.5,0.75,1),na.rm=TRUE)",sep="")))
  eval(parse(text=paste("purchaseddata['",col,"'] <- cut(purchaseddata$",col,", q,include.lowest = TRUE,labels = 1:4)",sep="")))
}


dep_vars <- c("A","C","D","F","G")
binaries <- as.data.frame(purchaseddata[,c("customer_ID","shopping_pt")])
names(binaries)<-c("customer_ID","shopping_pt")
dep_vars_binary <- c("B","E")
for(dep_var in dep_vars)  {
  temp <- as.data.frame(class.ind(purchaseddata[,c(paste(dep_var))]))
  names(temp)<-paste(dep_var,names(temp),sep="")
  dep_vars_binary <- c(dep_vars_binary,names(temp))
  binaries <- cbind(binaries,temp)
}
purchaseddata <- merge(purchaseddata,binaries,by=c("customer_ID","shopping_pt"))

sum.category <- NULL
for(col in cols)  {
  cat(col,"\n")
  coldf <- NULL
  for(dep_var in dep_vars_binary) {
    cat(dep_var,"-")
    eval(parse(text=paste("temp <- ddply(purchaseddata,.(",col,"),summarise,",dep_var," = mean(",dep_var,"))",sep="")))
    if(is.null(coldf))  {
      coldf <-temp
    } else {
      coldf = merge(coldf, temp, by=col)
    }
  }
  coldf['variable']<-col
  names(coldf)<-c('value',names(coldf)[2:length(names(coldf))])
  if(is.null(sum.category))  {
    sum.category<-coldf
  } else {
    sum.category = rbind(sum.category,coldf)
  }
  cat("\n")
}

sum.category <- sum.category %>%
                  select(variable,everything())

write.csv(sum.category,file="sum.category.csv")


# PREDICTION FUNCTIONS ---------------------------------------------------------------


load_data <- function() {
  setwd("E:/allstate")
  require(sqldf)
  require(MASS)
  require(dplyr)
  require(plyr)
  require(nnet)
  require(neuralnet)
  require(ROCR)
  require(randomForest)
  require(e1071)
  require(rpart)
  
  traindata <- read.csv("train.csv")
  traindata <- traindata %>% 
    arrange(customer_ID, shopping_pt) %>%
    group_by(customer_ID) %>%
    dplyr::mutate(rank=row_number())
  traindata <- as.data.frame(traindata)
  traindata['purchased_hour']<-substr(as.character(traindata$time),1,2)
  
  
  col_names <- c("A","C","D","F","G","day","purchased_hour","state","car_value","C_previous")
  binaries <- as.data.frame(traindata[,c("customer_ID","shopping_pt")])
  names(binaries)<-c("customer_ID","shopping_pt")
  
  for(col_name in col_names)  {
    temp <- as.data.frame(class.ind(traindata[,c(paste(col_name))]))
    names(temp)<-paste(col_name,names(temp),sep="")
    binaries <- cbind(binaries,temp)
  }
  
  
  #traindata <- traindata[ , -which(names(traindata) %in% col_names)]
  traindata <- merge(traindata,binaries,by=c("customer_ID","shopping_pt"))
  
  
  dep_vars <- c("A","B","C","D","E","F","G")
  Y_vars <-  c("B","E",names(traindata)[28:45])
  
  
  X_names <-c('car_age','risk_factor','age_oldest','age_youngest','group_size','married_couple','duration_previous','homeowner',
               'stateAL','stateAR','stateCO','stateCT','stateDC','stateDE','stateFL','stateGA','stateIA','stateID','stateIN',
               'stateKS','stateKY','stateMD','stateME','stateMO','stateMS','stateMT','stateND','stateNE','stateNH','stateNM',
               'stateNV','stateNY','stateOH','stateOK','stateOR','statePA','stateRI','stateSD','stateTN','stateUT','stateWA',
               'stateWI','stateWV','stateWY','car_valuea','car_valueb','car_valuec','car_valued','car_valuee',
               'car_valuef','car_valueg','car_valueh','car_valuei','C_previous1','C_previous2','C_previous3','C_previous4','car_age2','car_age3',
               paste('first3',Y_vars,sep = ""))
  
  X_names_logit <-c('car_age','risk_factor','age_oldest','age_youngest','group_size','married_couple','duration_previous','homeowner',
               'factor(state)','factor(car_value)','factor(C_previous)',paste('first3',Y_vars,sep = ""))
  
  purchaseddata <- traindata[traindata$record_type==1,]
  

  first3pct <- readRDS("first3pct.rds")
  purchaseddata <- merge(purchaseddata,first3pct,by=c("customer_ID"))
  purchaseddata$time <-NULL
  purchaseddata["car_age2"] <-purchaseddata$car_age*purchaseddata$car_age
  purchaseddata["car_age3"] <-purchaseddata$car_age2*purchaseddata$car_age
  purchaseddata$car_valueV1 <- NULL
  purchaseddata <<- cbind(purchaseddata[,! colnames(purchaseddata) %in% (X_names)],as.data.frame(scale(purchaseddata[,X_names])))
  traindata <<-traindata
  dep_vars <<- dep_vars
  Y_vars <<- Y_vars
  X_names <<-X_names
  X_names_logit <<- X_names_logit

}

gen_train_test_sets <- function() {
  
  purchase_train <- sample_frac(purchaseddata,0.7)
  sid <- unique(purchase_train$customer_ID)
  purchase_test <- purchaseddata[! (purchaseddata$customer_ID %in% sid),]
  purchase_train <<- purchase_train[is.finite(rowSums(purchase_train[,c(Y_vars,X_names)])),]
  purchase_test <<- purchase_test[is.finite(rowSums(purchase_test[,c(Y_vars,X_names)])),]
  
}

get_first3<-function()  {
  first3 <- traindata[traindata$rank<=3,]
  
  first3pct <- NULL
  for(col_name in Y_vars)  {
    eval(parse(text=paste("first3",col_name," <- ddply(first3,.(customer_ID),summarise,first3",col_name," = (sum(",col_name,",na.rm = TRUE)/3))",sep = "")))
    if (is.null(first3pct)) eval(parse(text=paste("first3pct<-first3",col_name,sep="")))
    else eval(parse(text=paste("first3pct<-merge(first3pct,first3",col_name,",by=c('customer_ID'))",sep="")))
  }
  saveRDS(first3pct,file = "first3pct.rds")
}


run_neuralnets <- function()  {
  for(dep_var in dep_vars)  {
    tryCatch({
      cat(dep_var,sep = "\n")
      y <- Y_vars[substr(Y_vars,1,1)==dep_var]
      nnformula <- as.formula(paste(paste(y,collapse='+'),' ~ ' ,paste(X_names,collapse='+')))
      eval(parse(text=paste("nn",dep_var," <- neuralnet(nnformula,purchase_train, hidden = c(5),linear.output = FALSE,threshold = 1,rep = 2,stepmax = 1e6)",sep="")))
      eval(parse(text=paste("save(nn",dep_var,",file='nn",dep_var,".rda')",sep="")))
      
      eval(parse(text=paste("nn",dep_var,"<-NULL",sep="")))
      gc()
    },error=function(e){})
  }
}
eval_neuralnets <- function() {
  acc_table <- NULL
  for(dep_var in dep_vars)  {
    tryCatch({
      cat(dep_var,sep = "\n")
      eval(parse(text=paste("load(file='nn",dep_var,".rda')",sep="")))
      
      eval(parse(text=paste("nn.prediction <- compute(nn",dep_var,",purchase_test[,X_names])",sep="")))
      nn.prediction <- as.data.frame(nn.prediction$net.result)
      names(nn.prediction) <- Y_vars[substr(Y_vars,1,1)==dep_var]
      if(length(names(nn.prediction))==1) {
        nn.prediction <- as.data.frame(cbind(ifelse(nn.prediction[dep_var] > 0.5,1,0),purchase_test[dep_var]))
      } else {
        nn.prediction <- as.data.frame(cbind(as.integer(substr(colnames(nn.prediction)[apply(nn.prediction,1,which.max)],2,2)),purchase_test[dep_var]))
      }
      names(nn.prediction)<-c("prediction","actual")
      confusion <- as.data.frame(table(nn.prediction$prediction,nn.prediction$actual))
      acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
      acc_table <- rbind(acc_table,c(dep_var,acc))
      
      nn.prediction <- as.data.frame(nn.prediction$prediction)
      names(nn.prediction)<- c(paste("nnpred",dep_var,sep = ""))
      purchase_test <<- cbind(purchase_test,nn.prediction)
      
      eval(parse(text=paste("nn",dep_var,"<-NULL",sep="")))
      gc()
    },error=function(e){})
  }
  return (acc_table)
}

run_logits <- function()  {
  for(dep_var in Y_vars)  {
    tryCatch({
      cat(dep_var,sep = "\n")
      #y <- Y_vars[substr(Y_vars,1,1)==dep_var]
      nnformula <- as.formula(paste(dep_var,' ~ ' ,paste(X_names,collapse='+')))
      eval(parse(text=paste("logit",dep_var," <- glm(formula=nnformula,family = binomial(link = 'logit'),data=purchase_train, control = list(maxit = 50))",sep="")))
      eval(parse(text=paste("save(logit",dep_var,",file='logit",dep_var,".rda')",sep="")))
      
      eval(parse(text=paste("logit",dep_var,"<-NULL",sep="")))
      gc()
    },error=function(e){})
  }
}
eval_logits <- function() {
  acc_table <- NULL
  predictions <- NULL
  for(dep_var in Y_vars)  {
    tryCatch({
      cat(dep_var,sep = "\n")
      eval(parse(text=paste("load(file='logit",dep_var,".rda')",sep="")))
      
      eval(parse(text=paste("logit.prediction <- predict(logit",dep_var,",newdata = purchase_test,type = 'response')",sep=""))) #predict(logitB,newdata = purchase_test,type = 'response')
      if(is.null(predictions))  {
        predictions <- as.data.frame(logit.prediction)
      } else {
        predictions <- cbind(predictions, as.data.frame(logit.prediction))  
      }
      
      logit.prediction <- ifelse(logit.prediction > 0.5,1,0)
      misClasificError <- mean(logit.prediction != purchase_test[dep_var],na.rm = TRUE)
      cat(1-misClasificError,"\n")
      acc_table <- rbind(acc_table,c(dep_var,1-misClasificError))
      
      eval(parse(text=paste("logit",dep_var,"<-NULL",sep="")))
      gc()
    },error=function(e){})
  }
  names(predictions) <- Y_vars
  for(dep_var in dep_vars)  {
    cat(dep_var,"\n")
    temp <- as.data.frame(predictions[,substr(colnames(predictions),1,1)==dep_var])
    
    if(length(names(temp))==1) {
      names(temp) <- dep_var
      temp <- as.data.frame(ifelse(temp[dep_var] > 0.5,1,0))
    } else {
      temp <- as.data.frame(as.integer(substr(colnames(temp)[apply(temp,1,which.max)],2,2)))
    }
    names(temp)<- c(paste("logit",dep_var,sep = ""))
    purchase_test <<- cbind(purchase_test,temp)
  }
  return (acc_table)
}

run_randomForests <- function()  {
  for(dep_var in dep_vars)  {
    tryCatch({
      cat(dep_var,sep = "\n")
      # y <- Y_vars[substr(Y_vars,1,1)==dep_var]
      rfformula <- as.formula(paste('as.factor(',dep_var,') ~ ' ,paste(X_names,collapse='+')))
      eval(parse(text=paste("rf",dep_var," <- randomForest(rfformula,purchase_train)",sep="")))
      eval(parse(text=paste("save(rf",dep_var,",file='rf",dep_var,".rda')",sep="")))
      
      eval(parse(text=paste("rf",dep_var,"<-NULL",sep="")))
      gc()
    },error=function(e){})
  }
}
eval_randomForests <- function() {
  MeanDecreaseGini <<- NULL
  acc_table <- NULL
  for(dep_var in dep_vars)  {
    tryCatch({
      cat(dep_var,sep = "\n")
      eval(parse(text=paste("load(file='rf",dep_var,".rda')",sep="")))
      
      eval(parse(text=paste("rf.prediction <- predict(rf",dep_var,",purchase_test[,X_names])",sep="")))
      rf.prediction <- as.data.frame(cbind(rf.prediction,purchase_test[dep_var]))
      names(rf.prediction)<-c("prediction","actual")
      confusion <- as.data.frame(table(rf.prediction$prediction,rf.prediction$actual))
      acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
      acc_table <- rbind(acc_table,c(dep_var,acc))
      
      rf.prediction <- as.data.frame(rf.prediction$prediction)
      names(rf.prediction)<- c(paste("rfpred",dep_var,sep = ""))
      purchase_test <<- cbind(purchase_test,rf.prediction)
      
      eval(parse(text=paste("temp <- as.data.frame(importance(rf",dep_var,"))",sep="")))
      names(temp)<-c(dep_var)
      temp['variable'] <- rownames(temp)
      rownames(temp) <- NULL
      
      if(is.null(MeanDecreaseGini)) {
        MeanDecreaseGini <<- temp
      } else {
        MeanDecreaseGini <<- merge(MeanDecreaseGini,temp,by=c("variable"))
      }
      
      eval(parse(text=paste("rf",dep_var,"<-NULL",sep="")))
      gc()
    },error=function(e){})
    
  }
  saveRDS(MeanDecreaseGini,file = "MeanDecreaseGini.rds")
  return (acc_table)
}

run_svm <- function(cost)  {
  for(dep_var in dep_vars)  {
    tryCatch({
      cat(dep_var,sep = "\n")
      rfformula <- as.formula(paste('as.factor(',dep_var,') ~ ' ,paste(X_names,collapse='+')))
      eval(parse(text=paste("svm",dep_var," <- svm(rfformula,data=purchase_train,cost=",cost,")",sep="")))
      eval(parse(text=paste("save(svm",dep_var,",file='svm",dep_var,".rda')",sep="")))
      
      eval(parse(text=paste("svm",dep_var,"<-NULL",sep="")))
      gc()
    },error=function(e){})
  }
}
eval_svm <- function()  {
  acc_table <- NULL
  for(dep_var in dep_vars)  {
    tryCatch({
      cat(dep_var,"\n")
      eval(parse(text=paste("load(file='svm",dep_var,".rda')",sep = "")))
      eval(parse(text=paste("svm.prediction <- predict(svm",dep_var,",purchase_test[,X_names])",sep="")))
      svm.prediction <- as.data.frame(cbind(svm.prediction,purchase_test[dep_var]))
      names(svm.prediction)<-c("prediction","actual")
      confusion <- as.data.frame(table(svm.prediction$prediction,svm.prediction$actual))
      acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
      cat(acc,"\n")
      acc_table <- rbind(acc_table,c(dep_var,acc))
      
      svm.prediction <- as.data.frame(svm.prediction$prediction)
      names(svm.prediction)<- c(paste("svmpred",dep_var,sep = ""))
      purchase_test <<- cbind(purchase_test,svm.prediction)
      
      eval(parse(text=paste("svm",dep_var,"<-NULL",sep="")))
      gc()
    },error=function(e){cat("error","\n")})
  }
  return (acc_table)
}



# RUN MODELS --------------------------------------------------------------

load_data()

cat("logit","\n")
run_logits()
tryCatch({eval_logits_acc<-eval_logits()},error=function(e){cat("error","\n")})

cat("nn","\n")
run_neuralnets()
tryCatch({eval_neuralnets_acc<-eval_neuralnets()},error=function(e){cat("error","\n")})

cat("rf","\n")
run_randomForests()
tryCatch({eval_randomForests_acc<-eval_randomForests()},error=function(e){cat("error","\n")})

cat("svm","\n")
run_svm()
tryCatch({eval_svm_acc<-eval_svm()},error=function(e){cat("error","\n")})

saveRDS(purchase_test,file="purchase_test_pred.rds")


# ANALYSE PREDICTED DATA --------------------------------------------------


rm(list = ls())
setwd("E:/allstate")


Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

predicted_data <- readRDS("purchase_test_pred.rds")
dep_vars <- c("A","B","C","D","E","F","G")
prefix <-c("logit","rfpred","nnpred","svmpred")

MeanDecreaseGin <- readRDS("MeanDecreaseGini.rds")
# car_age, age_youngest, age_oldest, florida and new york are important factors


# accuracy of individual models
acc_table <-NULL
for(dep_var in dep_vars) {
  cat(dep_var,"\n")
  for(p in prefix) {
    cat(p,"\n")
    confusion <- as.data.frame(table(predicted_data[,paste(p,dep_var,sep = "")],predicted_data[,dep_var]))
    acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
    acc_table <- rbind(acc_table,c(dep_var,p,acc))
    cat(dep_var,p,acc,"\n",sep = "-")
  }
}
write.csv(acc_table,"acc_table.csv")

# overall prediction- individual models
predicted_data['purchasedquote']<-paste(predicted_data$A,predicted_data$B,predicted_data$C,predicted_data$D,predicted_data$E,predicted_data$F,predicted_data$G,sep = "")
predicted_data['predictedquote_indmodels']<-paste(predicted_data$rfpredA,predicted_data$rfpredB,predicted_data$rfpredC,
                                                  predicted_data$rfpredD,predicted_data$rfpredE,predicted_data$svmpredF,
                                                  predicted_data$svmpredG,sep = "")
predicted_data['pred_vs_purch_indmodels']<- (predicted_data$purchasedquote==predicted_data$predictedquote_indmodels)*1



# Voting Model

for(dep_var in dep_vars) {
  t <- predicted_data[,paste(prefix,dep_var,sep="")]
  t = apply(t, 2, function(x) as.numeric(as.character(x)))
  predicted_data[paste('predicted_mode_',dep_var,sep = "")] = apply(t,1,function(x) Mode(c(x[1],x[2],x[3],x[4])))
}

# accuracy of voting model
acc_mode <- NULL
for(dep_var in dep_vars) {
  confusion <- as.data.frame(table(predicted_data[,paste('predicted_mode_',dep_var,sep = "")],predicted_data[,dep_var]))
  acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
  acc_mode <- rbind(acc_mode,c(dep_var,acc))
  
}

# overall prediction voting model
predicted_data['purchasedquote']<-paste(predicted_data$A,predicted_data$B,predicted_data$C,predicted_data$D,predicted_data$E,predicted_data$F,predicted_data$G,sep = "")
predicted_data['predictedquote']<-paste(predicted_data$predicted_mode_A,predicted_data$predicted_mode_B,predicted_data$predicted_mode_C,
                                        predicted_data$predicted_mode_D,predicted_data$predicted_mode_E,predicted_data$predicted_mode_F,
                                        predicted_data$predicted_mode_G,sep = "")
predicted_data['pred_vs_purch']<- (predicted_data$purchasedquote==predicted_data$predictedquote)*1

for (i in 1:7)  {
  acc1 <- mean((substr(predicted_data$purchasedquote,i,i)==substr(predicted_data$predictedquote,i,i))*1)
  acc2 <- mean((substr(predicted_data$secondquote,i,i)==substr(predicted_data$predictedquote,i,i))*1)
  cat(dep_vars[i],acc1,acc2,"\n")
  
}



# accuracy by different categories
summary(predicted_data$car_age)
q <- quantile(predicted_data$car_age,c(0,0.25,0.5,0.75,1))
predicted_data['car_age_q'] <- cut(predicted_data$car_age, q,include.lowest = TRUE,labels = 1:4)

summary(predicted_data$age_youngest)
q <- quantile(predicted_data$age_youngest,c(0,0.25,0.5,0.75,1))
predicted_data['age_youngest_q'] <- cut(predicted_data$age_youngest, q,include.lowest = TRUE,labels = 1:4)

summary(predicted_data$age_oldest)
q <- quantile(predicted_data$age_oldest,c(0,0.25,0.5,0.75,1))
predicted_data['age_oldest_q'] <- cut(predicted_data$age_oldest, q,include.lowest = TRUE,labels = 1:4)

predicted_data['state_cat'] <-ifelse(predicted_data$state=="FL","FL",ifelse(predicted_data$state=="NY","NY","OS"))



acc_table_cat <- NULL
keyvars <- c("car_age_q","age_youngest_q","age_oldest_q","state","risk_factor","married_couple","homeowner","car_value")
for(keyvar in keyvars)  {
  #cat(keyvar,"\n")
  for(q in unique(predicted_data[keyvar])[1:nrow(unique(predicted_data[keyvar])),1])  {
    #cat(q,"\n")
    temp <- predicted_data[predicted_data[keyvar]==q,]
    for(dep_var in dep_vars) {
     for(p in prefix) {
       confusion <- as.data.frame(table(temp[,paste(p,dep_var,sep = "")],temp[,dep_var]))
       confusion <- as.data.frame(apply(confusion, 2, function(x) as.numeric(as.character(x))))
       acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
       acc_table_cat <- rbind(acc_table_cat,c(keyvar,q,dep_var,p,acc))
     }
      #cat("\n")
    }
  }
}
acc_table_cat <- as.data.frame(acc_table_cat)
names(acc_table_cat)<-c("variable","varvalue","dep_var","model","acc")
acc_table_cat$acc <- as.numeric(as.character(acc_table_cat$acc))



# PREDICTION USING 2ND QUOTE ----------------------------------------------

rm(list = setdiff(ls(), lsf.str()))


load_data()

# filter only second quote. descriptive analysis shows that all customers have at least two quotes before purchasing
secondquote <- traindata[traindata$record_type!=1 & traindata$rank==2,]
secondquote['secondquote']<-paste(secondquote$A,secondquote$B,secondquote$C,secondquote$D,secondquote$E,secondquote$F,secondquote$G,sep = "")
secondquote <- secondquote[,c("customer_ID","secondquote","day","purchased_hour")]
names(secondquote)<-c("customer_ID","secondquote","secondquoteday","secondquotehour")
secondquote$secondquotehour <- as.integer(secondquote$secondquotehour)

purchaseddata <- merge(purchaseddata,secondquote,by=c("customer_ID"))

purchaseddata['purchasedquote']<-paste(purchaseddata$A,purchaseddata$B,purchaseddata$C,purchaseddata$D,purchaseddata$E,purchaseddata$F,purchaseddata$G,sep = "")
purchaseddata['sameday'] <- (purchaseddata$day ==purchaseddata$secondquoteday)*1



# compare second quote to purchased quote
purchaseddata['secondvspurchased']<- (purchaseddata$secondquote==purchaseddata$purchasedquote)*1

#accuracy of using second quote as the final prediction
cat("Accuracy of second quote as the final prediction : ",mean(purchaseddata$secondvspurchased))

# use the same training and test samples used for earlier models to be consistent
predicted_data <- readRDS("purchase_test_pred.rds")
predicted_data['purchasedquote']<-paste(predicted_data$A,predicted_data$B,predicted_data$C,predicted_data$D,predicted_data$E,predicted_data$F,predicted_data$G,sep = "")
predicted_data['predictedquote_indmodels']<-paste(predicted_data$rfpredA,predicted_data$rfpredB,predicted_data$rfpredC,
                                                  predicted_data$rfpredD,predicted_data$rfpredE,predicted_data$svmpredF,
                                                  predicted_data$svmpredG,sep = "")
predicted_data <- merge(predicted_data,secondquote,by=c("customer_ID"))

acc_secondquote <- NULL
for (i in 1:7)  {
  acc1 <- mean((substr(predicted_data$purchasedquote,i,i)==substr(predicted_data$predictedquote_indmodels,i,i))*1)
  acc2 <- mean((substr(predicted_data$secondquote,i,i)==substr(predicted_data$predictedquote_indmodels,i,i))*1)
  acc3 <- mean((substr(predicted_data$secondquote,i,i)==substr(predicted_data$purchasedquote,i,i))*1)
  acc_secondquote <- rbind(acc_secondquote, c(dep_vars[i],acc1,acc2,acc3))
}

purchase_test <- purchaseddata[(purchaseddata$customer_ID %in% predicted_data$customer_ID),]
purchase_train <- purchaseddata[!(purchaseddata$customer_ID %in% purchase_test$customer_ID),]
purchase_test <- purchase_test[is.finite(rowSums(purchase_test[,c(Y_vars,X_names)])),]
purchase_train <- purchase_train[is.finite(rowSums(purchase_train[,c(Y_vars,X_names)])),]


#predicting how likely customers to use the second quote as the final policy

#logit
logitformula <- as.formula(paste('secondvspurchased ~ ' ,paste(c(X_names,'sameday'),collapse='+')))
logit <- glm(formula=logitformula,family = binomial(link = 'logit'),data=purchase_train)
logit.prediction <- predict(logit,newdata = purchase_test)
#use a threshold of 0.7 as precision is more important
logit.prediction <- ifelse(logit.prediction > 0.7,1,0)
logit.prediction <- as.data.frame(cbind(logit.prediction,purchase_test['secondvspurchased']))
names(logit.prediction)<-c("prediction","actual")
confusion <- as.data.frame(table(logit.prediction$prediction,logit.prediction$actual))
logit.acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
logit.precision <- sum(confusion[confusion$Var1==1 & confusion$Var2==1,'Freq'])/sum(confusion[confusion$Var1==1,]$Freq)

#random forest
rfformula <- as.formula(paste('factor(secondvspurchased) ~ ' ,paste(c(X_names),collapse='+')))
rf <- randomForest(rfformula,purchase_train)
rf.prediction <- predict(rf,purchase_test[,c(X_names)],type = 'prob')
rf.prediction <- ifelse(rf.prediction[,2] > 0.5,1,0)
rf.prediction <- as.data.frame(cbind(rf.prediction,purchase_test['secondvspurchased']))
names(rf.prediction)<-c("prediction","actual")
confusion <- as.data.frame(table(rf.prediction$prediction,rf.prediction$actual))
rf.acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
rf.precision <- sum(confusion[confusion$Var1==1 & confusion$Var2==1,'Freq'])/sum(confusion[confusion$Var1==1,]$Freq)
save(rf,file='rfsecondquote.rda')

svm <- svm(rfformula,data=purchase_train,cost=1.2)
svm.prediction <- predict(svm,purchase_test[,X_names],probability=TRUE)
svm.prediction <- ifelse(svm.prediction[,2] > 0.7,1,0)
svm.prediction <- as.data.frame(cbind(svm.prediction,purchase_test['secondvspurchased']))
names(svm.prediction)<-c("prediction","actual")
confusion <- as.data.frame(table(svm.prediction$prediction,svm.prediction$actual))
svm.acc <- sum(confusion[confusion$Var1==confusion$Var2,'Freq'])/sum(confusion$Freq)
svm.precision <- sum(confusion[confusion$Var1==1 & confusion$Var2==1,'Freq'])/sum(confusion[confusion$Var1==1,]$Freq)



predicted_data <- cbind(predicted_data,rf.prediction$prediction)
names(predicted_data)[180]<-"secondquote_prediction"

predicted_data['combined_prediction']<-ifelse(predicted_data$secondquote_prediction==1,predicted_data$secondquote,predicted_data$predictedquote_indmodels)

# compare combined_prediction to purchased quote
predicted_data['combinedvspurchased']<- (predicted_data$combined_prediction==predicted_data$purchasedquote)*1

#accuracy of using combined_prediction as the final prediction
cat("Accuracy of combined_prediction as the final prediction : ",mean(predicted_data[predicted_data$secondquote_prediction==1,]$combinedvspurchased))



# KAGGLE SUBMISSION CODE --------------------------------------------------
rm(list = ls())
setwd("E:/allstate")
require(sqldf)
require(MASS)
require(dplyr)
require(plyr)
require(nnet)
require(neuralnet)
require(ROCR)
require(randomForest)
require(e1071)
require(rpart)


test <- read.csv("test_v2.csv")


test <- test %>% 
  arrange(customer_ID, shopping_pt) %>%
  group_by(customer_ID) %>%
  dplyr::mutate(rank=row_number())
test <- as.data.frame(test)
test['purchased_hour']<-substr(as.character(test$time),1,2)


col_names <- c("A","C","D","F","G","day","purchased_hour","state","car_value","C_previous")
binaries <- as.data.frame(test[,c("customer_ID","shopping_pt")])
names(binaries)<-c("customer_ID","shopping_pt")

for(col_name in col_names)  {
  temp <- as.data.frame(class.ind(test[,c(paste(col_name))]))
  names(temp)<-paste(col_name,names(temp),sep="")
  binaries <- cbind(binaries,temp)
}


#traindata <- traindata[ , -which(names(traindata) %in% col_names)]
test <- merge(test,binaries,by=c("customer_ID","shopping_pt"))


secondquote <- test[test$record_type!=1 & test$rank==2,]
secondquote['secondquote']<-paste(secondquote$A,secondquote$B,secondquote$C,secondquote$D,secondquote$E,secondquote$F,secondquote$G,sep = "")
secondquote <- secondquote[,c("customer_ID","secondquote")]

dep_vars <- c("A","B","C","D","E","F","G")
Y_vars <-  c("B","E",names(test)[28:45])


X_names <-c('car_age','risk_factor','age_oldest','age_youngest','group_size','married_couple','duration_previous','homeowner',
            'stateAL','stateAR','stateCO','stateCT','stateDC','stateDE','stateFL','stateGA','stateIA','stateID','stateIN',
            'stateKS','stateKY','stateMD','stateME','stateMO','stateMS','stateMT','stateND','stateNE','stateNH','stateNM',
            'stateNV','stateNY','stateOH','stateOK','stateOR','statePA','stateRI','stateSD','stateTN','stateUT','stateWA',
            'stateWI','stateWV','stateWY','car_valuea','car_valueb','car_valuec','car_valued','car_valuee',
            'car_valuef','car_valueg','car_valueh','car_valuei','C_previous1','C_previous2','C_previous3','C_previous4','car_age2','car_age3',
            paste('first3',Y_vars,sep = ""))


# first3 <- test[test$rank<=3,]
# 
# first3pct <- NULL
# for(col_name in Y_vars)  {
#   eval(parse(text=paste("first3",col_name," <- ddply(first3,.(customer_ID),summarise,first3",col_name," = (sum(",col_name,",na.rm = TRUE)/length(customer_ID)))",sep = "")))
#   if (is.null(first3pct)) eval(parse(text=paste("first3pct<-first3",col_name,sep="")))
#   else eval(parse(text=paste("first3pct<-merge(first3pct,first3",col_name,",by=c('customer_ID'))",sep="")))
# }
# saveRDS(first3pct,file = "first3pct_test.rds")

first3pct <- readRDS("first3pct_test.rds")


test_unique <- test %>% group_by(customer_ID) %>% slice(which.max(rank))


test_unique <- merge(test_unique,first3pct,by=c("customer_ID"))
test_unique$time <-NULL
test_unique["car_age2"] <-test_unique$car_age*test_unique$car_age
test_unique["car_age3"] <-test_unique$car_age2*test_unique$car_age
test_unique <<- cbind(test_unique[,! colnames(test_unique) %in% (X_names)],as.data.frame(scale(test_unique[,X_names])))

saveRDS(test_unique,"test_uniqe.rds")

test_unique <- readRDS("test_uniqe.rds")
#test_unique[is.na(test_unique)] <- 0

load(file='rfsecondquote.rda')
rf.prediction <- predict(rf,test_unique[,c(X_names)],type = 'prob')
rf.prediction <- ifelse(rf.prediction[,2] > 0.5,1,0)
test_unique <- cbind(test_unique,rf.prediction)
names(test_unique)[148]<-"secondquote_prediction"
test_unique <- merge(test_unique,secondquote,by=c("customer_ID"))
test_unique['most_recent'] <- paste(test_unique$A,test_unique$B,test_unique$C,test_unique$D,test_unique$E,test_unique$F,test_unique$G,sep = "")

test_unique_complete <- test_unique[!is.na(test_unique$secondquote_prediction),]
test_unique_na <- test_unique[is.na(test_unique$secondquote_prediction),]

load(file='rfA.rda')
A_prediction <- predict(rfA,test_unique_complete[,X_names])
load(file='rfB.rda')
B_prediction <- predict(rfB,test_unique_complete[,X_names])
load(file='rfC.rda')
C_prediction <- predict(rfC,test_unique_complete[,X_names])
load(file='rfD.rda')
D_prediction <- predict(rfD,test_unique_complete[,X_names])
load(file='rfE.rda')
E_prediction <- predict(rfE,test_unique_complete[,X_names])
load(file='svmF.rda')
F_prediction <- predict(svmF,test_unique_complete[,X_names])
load(file='svmG.rda')
G_prediction <- predict(svmG,test_unique_complete[,X_names])

test_unique_complete <- cbind(test_unique_complete,A_prediction,B_prediction,C_prediction,D_prediction,E_prediction,F_prediction,G_prediction)
test_unique_complete['predictedquote_indmodels']<-paste(test_unique_complete$A_prediction,test_unique_complete$B_prediction,test_unique_complete$C_prediction,
                                                        test_unique_complete$D_prediction,test_unique_complete$E_prediction,test_unique_complete$F_prediction,
                                                        test_unique_complete$G_prediction,sep = "")


test_unique_complete['combined_prediction']<-paste("a",ifelse(test_unique_complete$secondquote_prediction==1,test_unique_complete$secondquote,test_unique_complete$predictedquote_indmodels),sep="")

test_unique_complete <- test_unique_complete[,!(colnames(test_unique_complete) %in% c('predictedquote_indmodels','A_prediction','B_prediction','C_prediction','D_prediction','E_prediction','F_prediction','G_prediction'))]

test_unique_na['combined_prediction']<- paste("a",test_unique_na$secondquote,sep = "")
  
test_unique <- rbind(test_unique_complete,test_unique_na)
write.csv(test_unique[,c('customer_ID','combined_prediction')],file="submission4.csv")


# ENSEMBLE Method ---------------------------------------------------------

load_data()

# use the same training and test samples used for earlier models to be consistent
predicted_data <- readRDS("purchase_test_pred.rds")


purchase_test <- purchaseddata[(purchaseddata$customer_ID %in% predicted_data$customer_ID),]
purchase_train <- purchaseddata[!(purchaseddata$customer_ID %in% purchase_test$customer_ID),]
purchase_test <- purchase_test[is.finite(rowSums(purchase_test[,c(Y_vars,X_names)])),]
purchase_train <- purchase_train[is.finite(rowSums(purchase_train[,c(Y_vars,X_names)])),]

rf.prediction <- predict(rf,purchase_train[,c(X_names)],type = 'prob')
rf.prediction <- ifelse(rf.prediction[,2] > 0.5,1,0)
purchase_train <- cbind(purchase_train,rf.prediction)
names(purchase_train)[147]<-"secondquote_prediction"

A_prediction <- predict(rfA,purchase_train[,X_names])
B_prediction <- predict(rfB,purchase_train[,X_names])
C_prediction <- predict(rfC,purchase_train[,X_names])
D_prediction <- predict(rfD,purchase_train[,X_names])
E_prediction <- predict(rfE,purchase_train[,X_names])
F_prediction <- predict(svmF,purchase_train[,X_names])
G_prediction <- predict(svmG,purchase_train[,X_names])
purchase_train <- cbind(purchase_train,A_prediction,B_prediction,C_prediction,D_prediction,E_prediction,F_prediction,G_prediction)

nn_cols <- c(X_names,'secondquote_prediction','A_prediction'             ,'B_prediction'             ,'C_prediction'             ,'D_prediction'
             ,'E_prediction'
             ,'F_prediction'
             ,'G_prediction')
factorsNumeric <- function(d) modifyList(d, lapply(d[, sapply(d, is.factor)], asNumeric))
asNumeric <- function(x) as.numeric(as.character(x))
temp <- factorsNumeric(purchase_train[, nn_cols]) 
purchase_train <- cbind(purchase_train[,Y_vars],temp)



for(dep_var in dep_vars)  {
    tryCatch({
      cat(dep_var,sep = "\n")
      y <- Y_vars[substr(Y_vars,1,1)==dep_var]
      nnformula <- as.formula(paste(paste(y,collapse='+'),' ~ ' ,paste(nn_cols,collapse='+')))
      eval(parse(text=paste("nnensem",dep_var," <- neuralnet(nnformula,purchase_train, hidden = c(10,10),linear.output = FALSE,threshold = 1,rep = 2,stepmax = 1e6)",sep="")))
      eval(parse(text=paste("save(nnensem",dep_var,",file='nn",dep_var,".rda')",sep="")))
    },error=function(e){})
  }


test_unique <- readRDS("test_uniqe.rds")
test_unique[is.na(test_unique)] <- 0

rf.prediction <- predict(rf,test_unique[,c(X_names)],type = 'prob')
rf.prediction <- ifelse(rf.prediction[,2] > 0.5,1,0)
test_unique <- cbind(test_unique,rf.prediction)
names(test_unique)[148]<-"secondquote_prediction"
A_prediction <- predict(rfA,test_unique[,X_names])
B_prediction <- predict(rfB,test_unique[,X_names])
C_prediction <- predict(rfC,test_unique[,X_names])
D_prediction <- predict(rfD,test_unique[,X_names])
E_prediction <- predict(rfE,test_unique[,X_names])
F_prediction <- predict(svmF,test_unique[,X_names])
G_prediction <- predict(svmG,test_unique[,X_names])
test_unique <- cbind(test_unique,A_prediction,B_prediction,C_prediction,D_prediction,E_prediction,F_prediction,G_prediction)
test_unique <- factorsNumeric(test_unique[, nn_cols]) 

for(dep_var in dep_vars)  {
  tryCatch({
    cat(dep_var,sep = "\n")
    #eval(parse(text=paste("load(file='nn",dep_var,".rda')",sep="")))
    
    eval(parse(text=paste("nn.prediction <- compute(nnensem",dep_var,",test_unique[,nn_cols])",sep="")))
    nn.prediction <- as.data.frame(nn.prediction$net.result)
    names(nn.prediction) <- Y_vars[substr(Y_vars,1,1)==dep_var]
    if(length(names(nn.prediction))==1) {
      nn.prediction <- as.data.frame(ifelse(nn.prediction[dep_var] > 0.5,1,0))
    } else {
      nn.prediction <- as.data.frame(as.integer(substr(colnames(nn.prediction)[apply(nn.prediction,1,which.max)],2,2)))
    }
    names(nn.prediction)<- paste("nnensam",dep_var,sep="")
    
  },error=function(e){})
}
