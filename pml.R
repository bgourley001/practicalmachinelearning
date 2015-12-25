#Practical Machine Learning Project

library(caret)
#library(doParallel)

#register cores for parallel operations
#registerDoParallel(cores = detectCores(all.tests = TRUE) - 2)

#if data does not exist, then get it from the url's
if(!file.exists("./training.csv")) {
    #get the training data
    url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    training <- read.csv(url(url))

    #write the training data
    write.csv(training, file = "./training.csv")

    #get the test data
    url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    testing <- read.csv(url(url))

    #write the test data
    write.csv(testing,file="./testing.csv")
}

#read in training and test data from working directory
training <- read.csv("./training.csv",header=TRUE)
testing <- read.csv("./testing.csv",header=TRUE)

dim(training)
dim(testing)

#clean up training dataset
#remove columns with NA#s and near-zero variances
training.clean <- training[,sapply(training,function(x)!any(is.na(x)))]
#examine near zero variances
nzv <- nearZeroVar(training.clean,saveMetrics = TRUE)
nzv
training.clean <- training.clean[,nzv$nzv==FALSE]
dim(training.clean)

#remove columns 1:7 as they have no predictive value
training.clean <- training.clean[,-c(1:7)]
dim(training.clean)

#split training.clean into train (60%) and validate (40%) data sets
inTrain <- createDataPartition(y = training.clean$classe,p=0.60,list=FALSE)
train <- training.clean[inTrain,]
validate <- training.clean[-inTrain,]
dim(train)
dim(validate)

set.seed(12345)

#examine high correlations
correlations <- abs(cor(train[,-53]))
diag(correlations) <- 0
high.correlations <- which(correlations > 0.8, arr.ind = T)
high.correlations.names <- names(train)[unique(high.correlations[,2])]
train.high.correlations <- train[high.correlations.names]
dim(train.high.correlations)
#run pca on train.high.correlations
trans = preProcess(train.high.correlations, 
                   method=c("BoxCox", "center", 
                            "scale", "pca"))
PC = predict(trans, train.high.correlations)

#build random forest model
if(!file.exists("./model.rf.RData")) {
    control <- trainControl(method = "repeatedcv",number = 10,repeats = 3,allowParallel = TRUE)
    model.rf <- train(classe ~ .,data=train,method="rf",prox=TRUE,trControl=control)
    #save model
    save(model.rf,file="./model.rf.RData")
} else {
    load(file="./model.rf.RData")
}
pred.rf <- predict(model.rf,validate)
validate$predCorrect <- pred.rf == validate$classe
table(pred.rf,validate$classe)

confusionMatrix(validate$classe,pred.rf)

#build boosting model
if(!file.exists("./model.gbm.RData")) {
    model.gbm <- train(classe ~ .,data=train,method="gbm",verbose=FALSE)
    save(model.gbm,file="./model.gbm.RData")
} else {
    load(file="./model.gbm.RData")
}

pred.gbm <- predict(model.gbm,validate)
validate$predCorrect <- pred.gbm == validate$classe
table(pred.gbm,validate$classe)

confusionMatrix(validate$classe,pred.gbm)

#use random forest model as final model as accuracy is greater
final.model <- model.rf

#submission code
pred.submission <- predict(model.rf,testing)
answers <- as.character(pred.submission)

pml_write_files <- function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)
