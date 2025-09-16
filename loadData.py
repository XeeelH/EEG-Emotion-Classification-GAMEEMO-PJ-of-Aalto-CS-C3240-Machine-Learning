import kagglehub

# Download latest version
path = kagglehub.dataset_download("sigfest/database-for-emotion-recognition-system-gameemo")

print("Path to dataset files:", path)
