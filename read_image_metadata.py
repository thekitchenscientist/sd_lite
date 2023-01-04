from PIL import Image
import os

image_folder = r'C:\Users\stabl\source\repos\sd_lite\outputs'

def get_filepaths(parentpath, filepaths):
    paths = []
    for path in filepaths:
        try:
            new_parent = os.path.join(parentpath, path)
            paths += get_filepaths(new_parent, os.listdir(new_parent))
        except NotADirectoryError:
            paths.append(os.path.join(parentpath, path))
    return paths

filepaths = get_filepaths(image_folder, os.listdir(image_folder))

#print(filepaths)

for path in filepaths:
    # This is obviously a flawed way to check for an image but this is just
    # a demo script anyway.
    if path[-6:] not in ("_1.png", "_1.jpg","_1.PNG", "_1.JPG"):
        continue
    img = Image.open(path).convert('RGB')
    img.load()  # Needed only for .png EXIF data (see citation above)
    print(img.info['prompt'])