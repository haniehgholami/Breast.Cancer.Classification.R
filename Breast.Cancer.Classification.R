library(tidyverse)
library(tidymodels)
library(MASS)
library(FNN)
library(pROC)
library(class)
library(ISLR)
library(naivebayes)

### Creating performance index by confusion matrix
pref_index = function(cm){
  overall_error_rate = 1 - sum(diag(cm))/sum(cm)
  accuracy = sum(diag(cm))/sum(cm)
  sensitivity = cm[2,2] / (cm[1,2] + cm[2,2])
  specificity = cm[1,1] / (cm[2,1] + cm[1,1])
  return(c(overall_error = overall_error_rate, accu = accuracy, sens = sensitivity, spec = specificity))
}

breast = read.csv("breast_cancer.csv")
view(breast)
glimpse(breast)
str(breast)
breast %>% count(Diagnosis)

breast$Diagnosis = factor(breast$Diagnosis, labels = c("No", "Yes"))
breast %>% count(Diagnosis)

set.seed(5)
training_index = sample(1:nrow(breast), nrow(breast) * 0.7, replace = F)
head(training_index)

d_train = breast[training_index,]
d_test = breast[-training_index,]
glimpse(d_train)
dim(d_train)
glimpse(d_test)
dim(d_test)

### KNN Tuning
k_vec = c(1, 3, 5, 7, 15, 20)
error_vec = c()
for(i in 1:length(k_vec)){
  KNN_model = knn(breast[training_index,-31],
                  breast[-training_index,-31],
                  d_train$Diagnosis,
                  k_vec[i])
  error_vec[i] = mean(KNN_model != d_test$Diagnosis)
}
data.frame(k_vec, error_vec)

### Estimate KNN with best k=7
knn_pred = knn(train = breast[training_index,-31],
               test = breast[-training_index,-31],
               cl = d_train$Diagnosis,
               k = 7)
table(knn_pred, d_test$Diagnosis)
pref_index(table(knn_pred, d_test$Diagnosis))

### Confusion matrix for KNN (Fixed variable name here)
knn_cm = table(pred = knn_pred, obs = d_test$Diagnosis)
pref_index(knn_cm)

##### Logistic regression
logreg = glm(Diagnosis ~ ., data = d_train, family = "binomial")
summary(logreg)

#### Ratio
exp(logreg$coefficients[4])
logreg_pred = predict.glm(logreg, newdata = d_test, type = "response")
head(logreg_pred)

#### If we want to extend threshold 
logreg_pred2 = ifelse(logreg_pred > 0.5, "Yes", "No")
head(logreg_pred2)
table(pred = logreg_pred2, obs = d_test$Diagnosis)
pref_index(table(pred = logreg_pred2, obs = d_test$Diagnosis))

#### LDA
ldaout = lda(Diagnosis ~ ., data = d_train)
ldaout
lda_pred = predict(ldaout, newdata = d_test)
names(lda_pred)
head(lda_pred$posterior)
head(lda_pred$class)
table(lda_pred$class, obs = d_test$Diagnosis)
pref_index(table(lda_pred$class, obs = d_test$Diagnosis))

#### QDA
qdaout = qda(Diagnosis ~ ., data = d_train)
qda_pred = predict(qdaout, newdata = d_test)
names(qda_pred)
head(qda_pred$class)
head(qda_pred$posterior)
table(pred = qda_pred$class, obs = d_test$Diagnosis)
pref_index(table(pred = qda_pred$class, obs = d_test$Diagnosis))

#### Naive Bayes
nbout = naive_bayes(Diagnosis ~ .,
                    data = d_train,
                    usekernel = TRUE)
summary(nbout)
nb_pred = predict(nbout, d_test)
head(nb_pred)

### Also we can predict this way if we want to compute ROC curve
nb_pred_prob = predict(nbout, d_test, type = "prob")
head(nb_pred_prob)
table(nb_pred, d_test$Diagnosis)
pref_index(table(nb_pred, d_test$Diagnosis))

#### Compare between all models with pref_index
#### KNN
table(knn_pred, d_test$Diagnosis)
pref_index(table(knn_pred, d_test$Diagnosis))

#### Logistic
table(logreg_pred2, obs = d_test$Diagnosis)
pref_index(table(logreg_pred2, obs = d_test$Diagnosis))

#### LDA
table(lda_pred$class, obs = d_test$Diagnosis)
pref_index(table(lda_pred$class, obs = d_test$Diagnosis))

### QDA
table(pred = qda_pred$class, obs = d_test$Diagnosis)
pref_index(table(pred = qda_pred$class, obs = d_test$Diagnosis))

#### NB
table(nb_pred, d_test$Diagnosis)
pref_index(table(nb_pred, d_test$Diagnosis))

### Choose best
round(pref_index(table(knn_pred, d_test$Diagnosis)), 3)      ### KNN
round(pref_index(table(logreg_pred2, obs = d_test$Diagnosis)), 3) ### Logistic
round(pref_index(table(lda_pred$class, obs = d_test$Diagnosis)), 3) ### LDA
round(pref_index(table(pred = qda_pred$class, obs = d_test$Diagnosis)), 3) ### QDA
round(pref_index(table(nb_pred, d_test$Diagnosis)), 3)     #### NB

##### Best k here is 7 and best model is LDA