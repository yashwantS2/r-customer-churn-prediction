# Load necessary libraries
install.packages("rpart")      # Decision tree model
install.packages("rpart.plot") # Plotting the decision tree
install.packages("caTools")    # For splitting the data
install.packages("caret")      # For confusionMatrix, precision, recall, F1
install.packages("dplyr")

library(rpart)
library(rpart.plot)
library(caTools)
library(caret)
library(dplyr)

# Load the dataset using file.choose()
df <- read.csv(file.choose())

# Drop the first column (assuming it's an index or unwanted column)
df <- df[, -1]

# View the data (optional, but helps in checking the structure)
View(df)

# Set target column and separate features (x) and target (y)
target_column <- "Churn"  # The target column 'Churn' with 1 (churn) and 0 (no churn)
x <- df %>% select(-all_of(target_column))  # Feature columns
y <- df[[target_column]]  # Target column

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
id <- sample(2, nrow(df), prob = c(0.7, 0.3), replace = TRUE)
train <- df[id == 1, ]
test <- df[id == 2, ]

# Separate x and y for train and test sets
x_train <- train %>% select(-all_of(target_column))
y_train <- train[[target_column]]

x_test <- test %>% select(-all_of(target_column))
y_test <- test[[target_column]]
#=======================LOAD TRAIN AND TEST DONE============================#

#=======================DECISION TREE============================#
# Train the Decision Tree Model
customerChurnModel <- rpart(Churn ~ ., data = train, method = "class")

# Plot the Decision Tree
rpart.plot(customerChurnModel, type = 3, extra = 101, fallen.leaves = TRUE)

# Make Predictions on the Test Set
predictions <- predict(customerChurnModel, x_test, type = "class")

# Evaluate Model Performance using Confusion Matrix
confusion <- confusionMatrix(predictions, factor(y_test))
print("Decision Tree Model - Confusion Matrix and Performance Metrics:")
print(confusion)

# Precision, Recall, F1 Score
dt_precision <- confusion$byClass['Pos Pred Value']
dt_recall <- confusion$byClass['Sensitivity']
dt_f1 <- (2 * dt_precision * dt_recall) / (dt_precision + dt_recall)
cat("Decision Tree - Precision:", dt_precision, "Recall:", dt_recall, "F1 Score:", dt_f1, "\n")
#=======================DECISION TREE DONE WITHOUT BALANCE============================#

#=================== BALANCED DATA USING ROSE ===================#

# Install ROSE package
install.packages("ROSE")
library(ROSE)

# Apply ROSE for balancing data (it handles both over-sampling and under-sampling)
train_balanced <- ovun.sample(Churn ~ ., data = train, method = "both", p = 0.5, seed = 123)$data

# Check class distribution after balancing
table(train_balanced$Churn)

# Re-separate x and y after SMOTE for balanced data
x_train_balanced <- train_balanced %>% select(-all_of(target_column))
y_train_balanced <- train_balanced[[target_column]]

# Train the Decision Tree model
customerChurnModel <- rpart(Churn ~ ., data = train_balanced, method = "class", 
                            control = rpart.control(maxdepth = 6, minsplit = 8))

# Plot the decision tree
rpart.plot(customerChurnModel, type = 3, extra = 101, fallen.leaves = TRUE)

# Make predictions on the test set
predictions <- predict(customerChurnModel, x_test, type = "class")

# Evaluate Model Performance using Confusion Matrix
confusion <- confusionMatrix(predictions, factor(y_test))
print("Balanced Decision Tree Confusion Matrix and Performance Metrics:")
print(confusion)

# Precision, Recall, F1 Score
dt_balanced_precision <- confusion$byClass['Pos Pred Value']
dt_balanced_recall <- confusion$byClass['Sensitivity']
dt_balanced_f1 <- (2 * dt_balanced_precision * dt_balanced_recall) / (dt_balanced_precision + dt_balanced_recall)
cat("Balanced Decision Tree - Precision:", dt_balanced_precision, "Recall:", dt_balanced_recall, "F1 Score:", dt_balanced_f1, "\n")

# =================== BALANCED DATA USING ROSE DONE===================#

#=======================LOGISTIC REGRESSION MODEL============================#
# Logistic Regression Model
logistic_model <- glm(Churn ~ ., data = train, family = binomial)

# Make predictions on the test set
logistic_predictions <- predict(logistic_model, x_test, type = "response")
logistic_class <- ifelse(logistic_predictions > 0.5, 1, 0)

# Evaluate Logistic Regression model
logistic_confusion <- confusionMatrix(factor(logistic_class), factor(y_test))
print("Logistic Regression Confusion Matrix and Performance Metrics:")
print(logistic_confusion)

# Precision, Recall, F1 Score
logistic_precision <- logistic_confusion$byClass['Pos Pred Value']
logistic_recall <- logistic_confusion$byClass['Sensitivity']
logistic_f1 <- (2 * logistic_precision * logistic_recall) / (logistic_precision + logistic_recall)
cat("Logistic Regression - Precision:", logistic_precision, "Recall:", logistic_recall, "F1 Score:", logistic_f1, "\n")
#=======================LOGISTIC REGRESSION MODEL DONE============================#

#=======================GRADIENT BOOSTING MODEL============================#
# Gradient Boosting Model
install.packages("gbm")
library(gbm)
gbm_model <- gbm(Churn ~ ., data = train, distribution = "bernoulli", n.trees = 100, interaction.depth = 3, shrinkage = 0.1, cv.folds = 5)

# Make predictions on the test set
gbm_predictions <- predict(gbm_model, x_test, n.trees = gbm_model$n.trees, type = "response")
gbm_class <- ifelse(gbm_predictions > 0.5, 1, 0)

# Evaluate Gradient Boosting model
gbm_confusion <- confusionMatrix(factor(gbm_class), factor(y_test))
print("Gradient Boosting Confusion Matrix and Performance Metrics:")
print(gbm_confusion)

# Precision, Recall, F1 Score
gbm_precision <- gbm_confusion$byClass['Pos Pred Value']
gbm_recall <- gbm_confusion$byClass['Sensitivity']
gbm_f1 <- (2 * gbm_precision * gbm_recall) / (gbm_precision + gbm_recall)
cat("Gradient Boosting - Precision:", gbm_precision, "Recall:", gbm_recall, "F1 Score:", gbm_f1, "\n")
#=======================GRADIENT BOOSTING MODEL DONE============================#

#=======================SUPPORT VECTOR MACHINE MODEL============================#
# Support Vector Machine (SVM) Model
install.packages("e1071")
library(e1071)
svm_model <- svm(Churn ~ ., data = train, kernel = "linear", probability = TRUE)

# Make predictions on the test set
svm_predictions <- predict(svm_model, x_test)
svm_class <- factor(ifelse(svm_predictions > 0.5, 1, 0))

# Evaluate SVM model
svm_confusion <- confusionMatrix(svm_class, factor(y_test))
print("Support Vector Machine (SVM) Confusion Matrix and Performance Metrics:")
print(svm_confusion)

# Precision, Recall, F1 Score
svm_precision <- svm_confusion$byClass['Pos Pred Value']
svm_recall <- svm_confusion$byClass['Sensitivity']
svm_f1 <- (2 * svm_precision * svm_recall) / (svm_precision + svm_recall)
cat("SVM - Precision:", svm_precision, "Recall:", svm_recall, "F1 Score:", svm_f1, "\n")
#=======================SUPPORT VECTOR MACHINE MODEL DONE============================#

#VISUALISING:---

# Install ggplot2 for visualization
install.packages("ggplot2")
library(ggplot2)

# Store the performance metrics in a data frame
results <- data.frame(
  Model = c("Decision Tree", "Balanced Decision Tree", "Logistic Regression", 
            "Gradient Boosting", "SVM"),
  Precision = c(dt_precision, dt_balanced_precision, logistic_precision, 
                gbm_precision, svm_precision),
  Recall = c(dt_recall, dt_balanced_recall, logistic_recall, gbm_recall, svm_recall),
  F1_Score = c(dt_f1, dt_balanced_f1, logistic_f1, gbm_f1, svm_f1)
)

# Convert data to long format for easier plotting with ggplot2
results_long <- tidyr::pivot_longer(results, cols = c(Precision, Recall, F1_Score), 
                                    names_to = "Metric", values_to = "Value")

# Plot Precision, Recall, and F1 Score for each model
ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Comparison of Model Performance Metrics", 
       x = "Model", y = "Score") +
  scale_fill_brewer(palette = "Set1") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#-----------------------------


# Extract precision data into a data frame
precision_results <- data.frame(
  Model = c("Decision Tree", "Balanced Decision Tree", "Logistic Regression", 
            "Gradient Boosting", "SVM"),
  Precision = c(dt_precision, dt_balanced_precision, logistic_precision, 
                gbm_precision, svm_precision)
)

# Plot Precision for each model
ggplot(precision_results, aes(x = Model, y = Precision, fill = Model)) + 
  geom_bar(stat = "identity") +
  labs(title = "Precision Comparison Across Models", x = "Model", y = "Precision") +
  scale_fill_brewer(palette = "Set3") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = round(Precision, 2)), vjust = -0.3)


#----------------------------

# Extract accuracy for each model from the confusion matrix results
dt_accuracy <- confusion$overall["Accuracy"]  # Decision Tree
dt_balanced_accuracy <- confusion_balanced$overall["Accuracy"]  # Balanced Decision Tree
logistic_accuracy <- logistic_confusion$overall["Accuracy"]  # Logistic Regression
gbm_accuracy <- gbm_confusion$overall["Accuracy"]  # Gradient Boosting
svm_accuracy <- svm_confusion$overall["Accuracy"]  # SVM


# Create a data frame with accuracy results
accuracy_results <- data.frame(
  Model = c("Decision Tree", "Balanced Decision Tree", "Logistic Regression", 
            "Gradient Boosting", "SVM"),
  Accuracy = c(dt_accuracy, dt_balanced_accuracy, logistic_accuracy, 
               gbm_accuracy, svm_accuracy)
)

# View the accuracy results
print(accuracy_results)


# Install ggplot2 if needed
install.packages("ggplot2")
library(ggplot2)

# Plot accuracy comparison for each model
ggplot(accuracy_results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.7) +
  labs(title = "Accuracy Comparison of Models", x = "Model", y = "Accuracy") +
  theme_minimal() +
  theme(legend.position = "none") +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.3)

#-------------------------------------

## Saving the model:

# Save the model in .rds format
saveRDS(customerChurnModel_balanced, file = "customerChurnModel_balanced.rds")

# Load the model
loaded_model <- readRDS("customerChurnModel_balanced.rds")

# Make predictions on the test data using the loaded model
predictions <- predict(loaded_model, x_test, type = "class")

# Evaluate the performance using a confusion matrix
confusion <- confusionMatrix(predictions, factor(y_test))
print(confusion)

