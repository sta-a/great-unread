
# ToDo on Cluster: set data dir paths in DH class in utils (2x), activate conda env, nr workers in s2v, copy utils into analysis folder
# src dir to cluster
# Single files to src dir on cluster
# Download whole s2v eng dir, excluding subdir simmxs (too big, not needed)

# Grid images only
# Download single files from cluster
# Download commands that contain cluster commands
# This command will recursively search for and delete all directories named mydir from the current directory and all its subdirectories.
find . -type d -name "mxcomb" # find all subdirs
find . -type d -name "*singleimage_cluster" -exec rm -rv {} +
find . -maxdepth 1 -type d ! -name "*singleimage*" ! -name "." -exec rm -rf {} + # delete all dirs that dont have singleimage in their name
find . -type d -name "mxcomb" -exec rm -rv {} +
find . -type d -name "mxeval" -exec rm -rv {} +
find . -type d -name "nkcomb" -exec rm -rv {} +
find . -type d -name "nkeval" -exec rm -rv {} +
# find . -type d -name "analysis" -exec rm -rv {} +
# find . -type d -name "analysis_s2v" -exec rm -rv {} +
find . -type f -name "mx_log*.txt" -delete
find . -type f -name "nk_log*.txt" -delete
find . -type f -name "nk_noedges.txt" -delete
# Download s2v dir
## From Cluster to harddrive
## From Computer to harddrive
# From harddrive to computer
# Harddrive to Cluster
# Download sh files from cluster
# Reupload src files if they have been deleted by cluster
# Reupload embeddings

























