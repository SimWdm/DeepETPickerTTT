import glob
import os
import shutil

def backup_python_files(src, dest, exclude_dirs=[]):
    """
    Recursively copies all Python files in src to dest. If dest or any subdirectory does not exist, they are created. This has saved my life quite a few times.
    """
    for file_path in glob.glob(os.path.join(src, '**', '*.py'), recursive=True):
        if "/code_backup/" in file_path:
            continue
        if any([file_path.startswith(dir) for dir in exclude_dirs]):
            continue
        new_path = f"{dest}/{file_path.replace('./', '')}"
        dirname = os.path.dirname(new_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy(file_path, new_path)
