# Thus far
#   - mathematics
# Next
#   - continue splitting

setwd()


# Functions
source("packages.R")


# Ensure that the required packages are available
packages()


# Libraries
library(caret)
library(data.table)
library(proxy)


# Data
dataurl <- 'https://raw.githubusercontent.com/exhypotheses/beans/develop/warehouse/data/modelling.csv'
beans <- data.table::fread(dataurl, header = TRUE, encoding = "UTF-8", 
                           data.table = TRUE, colClasses = c(class = "factor"))
str(beans)


# Random States
seed <- 5
base::set.seed(seed = seed)


# Frequencies
frequencies <- beans[, .N, by=class]


# Hence
training_fraction <- 0.6
start_fraction <-0.25

labels <- unique(beans$class)

X = data.table()
T = data.table()

for (label in labels) {
  
  
  # The members of <label>
  excerpt <- beans[label, on=.(class)]
  
  
  # The number of <label> instances
  N <- dim(excerpt)[1]
  
  
  # Therefore, the number of training instances will be
  n_training_points <- base::floor(training_fraction * N)
  
  
  # The initial number of spider points
  n_start_points <- base::floor(start_fraction * n_training_points)
  
  # A random selection of <n_start_points> spider points
  start_indices <- base::sample(x = 1:N, size = n_start_points)
  
  # The remaining points, which are the points the spiders can acquire
  nest <- excerpt[-start_indices,]
  acquisitors <- excerpt[start_indices,]
  
  selections <- caret::maxDissim(a = acquisitors, b = nest, 
                                 n = (n_training_points - n_start_points), obj = caret::minDiss())
  
  indices <- base::unique(c(selections, start_indices))
  left <- n_training_points - base::length(indices)
  outliers <- base::setdiff(1:N, indices)
  leftpoints <- base::sample(x = outliers, size = left, replace = FALSE)
  
  
  # The training & testing sets
  indices_ <- c(indices, leftpoints)
  training <- excerpt[indices_,]
  testing <- excerpt[-indices_,]
  
  
  # Finally
  X <- base::rbind(X, training)
  T <- base::rbind(T, testing)
  
}








