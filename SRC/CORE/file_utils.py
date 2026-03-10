import os


def remove_all_files_from_folder(folder_path: str):
    """
    Removes all files from the specified folder.

    Args:
        folder_path (str): The path to the folder from which to remove all files.

    Raises:
        ValueError: If the provided path is not a valid directory.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")

    # List all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file and delete it
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"Skipped (not a file): {file_path}")


if __name__ == "__main__":
    folder_to_clean = "/path/to/your/folder"
    remove_all_files_from_folder(folder_to_clean)
