SplitData <- function(x, training_fraction, initial_fraction){
  #' Strategically splits the data of a class into training & testing sets
  #' 
  #' Strategically splits the data of a class into training & testing sets.  It depends on
  #' caret::maxDissim
  #' 
  #' @param x A frame of data
  #' @param training_fraction The proportion of x required for model training
  #' @param initial_fraction The proportion of the -required # of training points- 
  #' that should be used to start the search for training points  
  
  
  # The instances of a data's class
  N <- dim(x)[1]
  reference <- 1:N
  
  
  # Therefore, the number of training points will be
  n_training_points <- base::floor(training_fraction * N)
  
  
  # And, the number of points that will initiate the training points search is
  n_initial_points <- base::floor(initial_fraction * n_training_points)
  
  
  # A random selection of <n_initial_points>
  start_indices <- base::sample(x = reference, size = n_initial_points)
  
  
  # Hence, the <base> points that initiate the search, and the <pool> points, from 
  # whence the remaining training points are selected
  base <- x[start_indices,]
  pool <- x[-start_indices,]
  
  
  #Selecting ...
  selections <- caret::maxDissim(a = base, b = pool, 
                                 n = (n_training_points - n_initial_points))
  
  # Each <base> point acquires a unique set of points from <pool>, but the 
  # intersection of <base> points acquisitions is not an empty set
  indices <- base::unique(c(selections, start_indices))
  n_short <- n_training_points - base::length(indices)
  unselected <- base::setdiff(reference, indices)
  remaining <- base::sample(x = unselected, size = n_short, replace = FALSE)
  
  
  # The training & testing sets
  i <- c(indices, remaining)
  training <- x[i,]
  testing <- x[-i,]
  
  return (list('training' <- training, 'testing' <-testing))
  
  
}
