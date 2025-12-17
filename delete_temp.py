import os
import glob

def delete_temp_files():
    # Get the current directory
    current_dir = os.getcwd()

    # Find all .tmp files in the current directory
    temp_files = glob.glob(os.path.join(current_dir, '*.tmp'))

    # Delete each temp file
    for file in temp_files:
        os.remove(file)
        #print(f"Deleted: {file}")

    print(f"Total temp files deleted: {len(temp_files)}")

if __name__ == "__main__":
    delete_temp_files()
