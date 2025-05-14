
#MSBA-635-77-4252: PREDICTIVE ANALYTICS-Spring 2025
#Assignment 2 Part 2 - Group based Pre-processing for Case Competition
#Amna Tariq, Lisa Shoff, Sibani Mohapatra


#The Preprocessing plan with team member assignments

#1)	Replace strange marital status values with “Other”   - Amna
#2)	Change all factors columns to factor – Education, Marital Status (wait until after we fix the weird values) Accepted Campaigns 1-5, Complain, Response - Lisa
#3)	Search the file for duplicate rows - Sibani
#4)	Impute the mean for Income rows with NA - Amna
#5)	DT Customer – group into years and maybe months – Lisa
#6)	Rename DT Customer and Recency for clarity - lisa
#7)	Create new column adding together all the Amount Spent Columns for “Total Amount Spent” - Sibani
#8)	Think about dummy variables - Amna
#9)	Combined column for “has accepted any previous campaign” – Lisa



library(caret)
library(tidyverse)
options(scipen=999)
setwd("C:/Users/tamna/Downloads")


havendata <- read.csv(file = "Gourmet Haven case data.csv", header=T)

colSums(is.na(havendata))

library(skimr)
skim(havendata) 


# 1. #Cleaning up the Marital Status column
havendata$Marital_Status <- replace(havendata$Marital_Status, 
                                    havendata$Marital_Status %in% c("YOLO", "Absurd", "Alone"), 
                                      "Other")
#Checking it worked
sort(table(havendata$Marital_Status))


# 2. Making the categorical columns into factors
havendata <- havendata %>%
  mutate(across(c(Education, Marital_Status, AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, Complain, Response), as.factor))

str(havendata)

#3 - Remove duplicate rows
havendata<- havendata[!duplicated(havendata), ]



# 5. DT Customer - clean up and add some grouped columns


library(lubridate)
havendata$Dt_Customer <- as.Date(havendata$Dt_Customer, format = "%m/%d/%Y")
havendata$WeeksSinceEnrollment <- as.integer(difftime(Sys.Date(), havendata$Dt_Customer, units = "weeks"))
havendata$MonthsSinceEnrollment <- interval(havendata$Dt_Customer, Sys.Date()) / months(1)
#havendata$Year_Enrolled <- year(havendata$Dt_Customer)
#havendata$Month_Enrolled <- month(havendata$Dt_Customer, label = TRUE, abbr = FALSE)
#havendata$Month_Enrolled <- factor(havendata$Month_Enrolled, levels = month.name, ordered = FALSE)


# 6. Renaming Columns for Clarity
names(havendata)[names(havendata) == "Dt_Customer"] <- "Date_Enrolled"
names(havendata)[names(havendata) == "Recency"] <- "Days_Since_Last_Purchase"

str(havendata)

#havendata$Days_Since_Last_Purchase_Buckets <- cut(havendata$Days_Since_Last_Purchase, 
                                breaks = seq(0, max(havendata$Days_Since_Last_Purchase, na.rm = TRUE) + 25, by = 25), 
                                include.lowest = TRUE, 
                                right = FALSE, 
                                labels = FALSE) 

#havendata$Days_Since_Last_Purchase_Buckets <- as.factor(havendata$Days_Since_Last_Purchase_Buckets)
#havendata <- havendata[, !(names(havendata) %in% "Days_Since_Last_Purchase")]



str(havendata)
table(havendata$Days_Since_Last_Purchase_Buckets)



# 7. Create new column adding together all the Amount spent Columns
havendata$TotalAmt_Spent <- havendata$MntWines +
  havendata$MntFruits +
  havendata$MntMeatProducts +
  havendata$MntFishProducts +
  havendata$MntSweetProducts


# 9. Creating Any_Campaign_Accepted Column
Accepted_Columns <- c("AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5")
havendata$Any_Campaign_Accepted <- rowSums(havendata[Accepted_Columns] == 1) > 0
havendata$Any_Campaign_Accepted <- as.integer(havendata$Any_Campaign_Accepted)

str(havendata)

# 10. Adding decade column for birth year
#havendata$Decade_Born <- floor(havendata$Year_Birth / 10) * 10
#havendata$Decade_Born <- as.factor(havendata$Decade_Born)
str(havendata)

dummies_model<-dummyVars(~.-Response,data=havendata)
havendata_predictors_dummy<- data.frame(predict(dummies_model, newdata = havendata))
havendata<- cbind(Response=havendata$Response, havendata_predictors_dummy)


# 4. Impute median income for missing income rows
#MOVING THIS TO THE END BECAUSE I FEEL LIKE IT NEEDS TO BE LAST 

havendata$Response<-as.factor(havendata$Response)
havendata$Response<-fct_recode(havendata$Response, No = "0", Yes = "1")
havendata$Response<-relevel(havendata$Response,ref="Yes")


set.seed(97)
index <- createDataPartition(havendata$Response, p = .8,list = FALSE)
havendata_train <- havendata[index,]
havendata_test <- havendata[-index,]



# Impute Mean of income
preProcess_missingdata_model <- preProcess(havendata_train, method='medianImpute')
preProcess_missingdata_model
havendata_train <-predict(preProcess_missingdata_model,newdata=havendata_train)
havendata_test <-predict(preProcess_missingdata_model,newdata=havendata_test)




#RUNNING A BACKWARD SELECTION MODEL - didnt get great stuff from here

#havendata_backwardmodel <- train(Response ~ .,
                      data = havendata_train,
                      method = "glmStepAIC",
                      direction="backward",
                      trControl =trainControl(method = "none",
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                      metric="ROC")


#coef(havendata_backwardmodel$finalModel)


#predprob_backward<-predict(havendata_backwardmodel , havendata_test, type="prob")
#library(ROCR)

#pred_backward <- prediction(predprob_backward$Yes, havendata_test$Response,label.ordering =c("No","Yes") )
#perf_backward <- performance(pred_backward, "tpr", "fpr")
#plot(perf_backward, colorize=TRUE)



#RUNNING AN XGBOOST TO SEE WHAT WE GET
install.packages("xgboost")
library(xgboost)
set.seed(8)
model_gbm <- train(Response ~ .,
                   data = havendata_train,
                   method = "xgbTree",
                   trControl =trainControl(method = "cv", 
                                           number = 5),
                   # provide a grid of parameters
                   tuneGrid = expand.grid(
                     nrounds = c(50,200),
                     eta = c(0.025, 0.05),
                     max_depth = c(2, 3),
                     gamma = 0,
                     colsample_bytree = 1,
                     min_child_weight = 1,
                     subsample = 1),
                   verbose=FALSE)


plot(model_gbm)
model_gbm$bestTune
plot(varImp(model_gbm))

str(havendata_train)

install.packages("SHAPforxgboost")
library(SHAPforxgboost)
str(Xdata)
Xdata<-as.matrix(select(havendata_train,-Response)) # change data to matrix for plots
# Crunch SHAP values
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

# SHAP importance plot
shap.plot.summary(shap)




xgboost_prob<- predict(model_gbm, havendata_test, type="prob")


#Step 4: Evaluate Model Performance
library(ROCR)
#In label.ordering the negative class is first then the positive class
pred = prediction(xgboost_prob$Yes, havendata_test$Response,label.ordering =c("No","Yes")) 

perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)


#Area under the curve is given by (do not worry about the syntax here):
  
  unlist(slot(performance(pred, "auc"), "y.values"))
  

# Use 4 most important predictor variables
top4<-shap.importance(shap, names_only = TRUE)[1:4]

for (x in top4) {
  p <- shap.plot.dependence(
    shap, 
    x = x, 
    color_feature = "auto", 
    smooth = FALSE, 
    jitter_width = 0.01, 
    alpha = 0.4
  ) +
    ggtitle(x)
  print(p)
}

