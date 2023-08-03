# Install and load required packages
#install.packages("quanteda")
# library(quanteda)

# # Set the directory where the CSV files are located
# csv_dir <- "/home/annina/scripts/great_unread_nlp/data/ngram_counts/eng"

# # Get a list of all CSV files in the directory
# csv_files <- list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE)

# # Create an empty list to store the delta results for each file
# all_delta_results <- list()

# # Iterate through each CSV file
# for (csv_file in csv_files) {
#   print(csv_file)
#   # Read the document-term matrix from the CSV file
#   dtm <- read.csv(csv_file)
  # print(names(dtm))
  # Check row names
  # print("Row Names:")
  # print(row.names(dtm))

  # # Check column names
  # print("Column Names:")
  # print(colnames(dtm))


  # any_missing <- any(is.na(dtm))
  # print(any_missing)
  # # # Convert the document-term matrix to a quanteda dfm object
  # dfm <- as.dfm(dtm)
  # print(dtm)
  
  # # Create a quanteda corpus from the dfm
  # corpus <- corpus(dtm, text_field = "file_name")
  
  # # Calculate delta measures
  # delta_measures <- c("burrows_delta", "zeta", "eders_delta", "cosine_delta",
  #                     "quadratic_delta", "pearsons_delta", "likelihood_delta",
  #                     "log_likelihood_delta", "chi_square_delta", "jensen_shannon_delta")
  
  # delta_results <- list()
  # for (measure in delta_measures) {
  #   deltas <- textstat_dist(corpus, method = measure)
  #   delta_results[[measure]] <- deltas
  # }
  
  # # Store the delta results for the current file in the list
  # all_delta_results[[csv_file]] <- delta_results
  
  # # Create a new filename with "quanteda-" added at the beginning
  # output_file <- paste0("quanteda-", basename(csv_file))
  
  # # Save each delta matrix as a separate CSV file
  # for (measure in delta_measures) {
  #   delta_matrix <- delta_results[[measure]]
  #   # Replace missing values (NA) with 0
  #   delta_matrix[is.na(delta_matrix)] <- 0
  #   output_path <- file.path("/home/annina/scripts/great_unread_nlp/src/test", paste0(output_file, "-", measure, ".csv"))
  #   write.table(delta_matrix, file = output_path, sep = ",", col.names = NA)
  # }
# }


# -----------------------------------------------------------

# Install and load required packages
# install.packages("stylo")
library(stylo)
languages <- c("eng", "ger")

  for (language in languages) {
  # Set the directory where the CSV files are located
  csv_dir <- file.path("/home/annina/scripts/great_unread_nlp/data/ngram_counts", language)
  print(csv_dir)

  # Get a list of all CSV files in the directory
  csv_files <- list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE)
  # Create an empty list to store the delta results for each file
  print(csv_files)

  all_delta_results <- list()
  # Iterate through each CSV file
  for (csv_file in csv_files) {

    # Read the document-term matrix from a CSV file (replace with your file path)
    dtm <- read.csv(csv_file, row.names = 1, check.names = FALSE)


    # Calculate delta measures
    delta_measures <- c("burrows", 
    # "zeta", 
    "linear", 
    "eder", 
    "edersimple")
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
        delta_results[[delta_measure]] <- as.matrix(dist.delta(dtm))
      # } else if (delta_measure == "zeta") {
      #   delta_results[[delta_measure]] <- delta_zeta(dtm)
      } else if (delta_measure == "linear") {
        delta_results[[delta_measure]] <- as.matrix(dist.argamon(dtm))
      } else if (delta_measure == "eders") {
        delta_results[[delta_measure]] <- as.matrix(dist.eder(dtm))
      } else if (delta_measure == "ederssimple") {
        delta_results[[delta_measure]] <- as.matrix(dist.simple(dtm))
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

    all_delta_results[[csv_file]] <- delta_results
    output_dir <- file.path("/home/annina/scripts/great_unread_nlp/data/distance", language, "stylo")
    dir.create(output_dir, showWarnings = FALSE)

    # Extract the number using regular expression
    number <- as.numeric(gsub("[^0-9]", "", basename(csv_file)))
    print(number)

    # Save each delta matrix as a separate CSV file
    for (measure in delta_measures) {
      delta_matrix <- delta_results[[measure]]
      # Replace missing values (NA) with 0
      #delta_matrix[is.na(delta_matrix)] <- 0
      output_path <- file.path(output_dir, paste0("dist-", measure, "-", number, ".csv"))
      write.table(delta_matrix, file = output_path, sep = ",", col.names = NA)
    }
  }
}