
# Functions
source("functions/ImportPackages.R")
source("functions/SplitData.R")


# Ensure that the required packages are available
ImportPackages()


# Libraries
library(caret)
library(data.table)
library(proxy)


# Data
dataurl <- 'https://raw.githubusercontent.com/exhypotheses/beans/develop/warehouse/data/modelling.csv'
beans <- data.table::fread(dataurl, header = TRUE, encoding = "UTF-8", 
                           data.table = TRUE, colClasses = c(class = "factor"))
str(beans)
frequencies <- beans[, .N, by=class]


# Random States
seed <- 5
base::set.seed(seed = seed)


# Hence
training_fraction <- 0.6
initial_fraction <-0.35


# The distinct classes of the data set
labels <- unique(beans$class)


# Hence
X = data.table()
T = data.table()

for (label in labels) {
  
  # The members of <label>
  excerpt <- beans[label, on=.(class)]
  
  # Split the data into training & testing sets
  splits <- SplitData(x = excerpt, training_fraction = training_fraction, 
                      initial_fraction = initial_fraction)
  
  # Finally
  X <- base::rbind(X, splits$training)
  T <- base::rbind(T, splits$testing)
  
}


# Save
data.table::fwrite(x = X, file = 'training.csv', row.names = FALSE, col.names = TRUE)
data.table::fwrite(x = T, file = 'testing.csv', row.names = FALSE, col.names = TRUE)





