import glob
import os
import shutil

def backup_python_files(src, dest, exclude_dirs=[], extensions=[".py", ".sh"]):
    """
    Recursively copies all Python files in src to dest. If dest or any subdirectory does not exist, they are created. This has saved my life quite a few times.
    """
    # find all files with the given extensions
    file_paths = []
    for ext in extensions:
        file_paths.extend(glob.glob(f"{src}/**/*{ext}", recursive=True))
    for file_path in file_paths:
        # skip backup files and excluded dirs
        if "/code_backup/" in file_path:
            continue
        if any([file_path.startswith(dir) for dir in exclude_dirs]):
            continue
        new_path = f"{dest}/{file_path.replace('./', '')}"
        dirname = os.path.dirname(new_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy(file_path, new_path)
