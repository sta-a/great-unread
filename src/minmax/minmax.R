# Install and load required packages
# install.packages("stylo")
library(stylo)
languages <- c("eng") #, "ger" ##########################

  for (language in languages) {
  csv_files <- c("/home/annina/scripts/great_unread_nlp/src/cluster/minmax.csv")

  

  # Iterate through each CSV file
  for (csv_file in csv_files) {

    # Read the document-term matrix from a CSV file (replace with your file path)
    dtm <- read.csv(csv_file, row.names = 1, check.names = FALSE)
    print(dtm)


    # Calculate delta measures
    delta_measures <- c("minmax")
    # "argamon", 
    # "eder",
    # "edersimple"
    # "zeta", 
    # "canberra"
    # "cosine", 
    # "quadratic",
    # "pearsons", 
    # "likelihood", 
    # "loglikelihood", 
    # "chi_square", 
    # "jensen_shannon"

    delta_results <- list()
    # Loop through each delta measure and calculate it for the given document term matrix
    for (delta_measure in delta_measures) {
      if (delta_measure == "burrows") {
        delta_results[[delta_measure]] <- as.matrix(dist.delta(dtm, scale = TRUE))
      } else if (delta_measure == "minmax") {
        delta_results[[delta_measure]] <- as.matrix(dist.minmax(dtm))
      # } else if (delta_measure == "zeta") {
      #   delta_results[[delta_measure]] <- delta_zeta(dtm)
      # } else if (delta_measure == "dist.canberra") {
      #   delta_results[[delta_measure]] <- as.matrix(dist.canberra(dtm))
      #} else if (delta_measure == "cosine") {
      #   delta_results[[delta_measure]] <- delta_cosine(dtm)
      # } else if (delta_measure == "quadratic") {
      #   delta_results[[delta_measure]] <- delta_quadratic(dtm)
      # } else if (delta_measure == "pearsons") {
      #   delta_results[[delta_measure]] <- delta_pearsons(dtm)
      # } else if (delta_measure == "likelihood") {
      #   delta_results[[delta_measure]] <- delta_likelihood(dtm)
      # } else if (delta_measure == "loglikelihood") {
      #   delta_results[[delta_measure]] <- delta_loglikelihood(dtm)
      # } else if (delta_measure == "chi_square") {
      #   delta_results[[delta_measure]] <- delta_chi_square(dtm)
      # } else if (delta_measure == "jensen_shannon") {
      #   delta_results[[delta_measure]] <- delta_jensen_shannon(dtm)
      }
    }

    # Save each delta matrix as a separate CSV file
    for (measure in delta_measures) {
      # print(measure)
      delta_matrix <- delta_results[[measure]]
      # print(delta_matrix[1:2, 1:2])
      # Replace missing values (NA) with 0
      #delta_matrix[is.na(delta_matrix)] <- 0
      print(delta_matrix)
    }
  }
}
