import os
import json
import re

def path_win2lin(path):
    return path.replace("\\", "/")

def path_valid(path):
    """ 
    所有图片复制到了 jsons/../images/
    有的路径不一定是 ../images/xxx.jpg 可能是 ../images/xxx/yyy.jpg (or png)
    因此使用正则表达式检查是否是 ../images/xxx.jpg 或 ../images/xxx.png，不允许有多级目录
    """
    pattern = r"^\.\./images/[^/]+\.jpg$|^\.\./images/[^/]+\.png$"
    if re.match(pattern, path):
        return True
    return False

def path_correction(path):
    """ 
    将 ../images/xxx/yyy.jpg 或 xxx.jpg 转换为 ../images/xxx.jpg (or png)
    例如
    000304.png -> ../images/000304.png
    ../images/220/000220.png -> ../images/000220.png
    any/path/xxx.jpg -> ../images/xxx.jpg
    """
    exs = [
        r"^.*\/([^/]+\.jpg)$", # any/path/xxx.jpg -> ../images/xxx.jpg
        r"^.*\/([^/]+\.png)$", # any/path/xxx.png -> ../images/xxx.png
        r"^([^/]+\.jpg)$", # xxx.jpg -> ../images/xxx.jpg
        r"^([^/]+\.png)$", # xxx.png -> ../images/xxx.png
    ]
    for ex in exs:
        match = re.match(ex, path)
        if match:
            filename = match.group(1)
            return f"../images/{filename}"
    return path

json_path = "datasets/20260513_drone/json"

fns = os.listdir(json_path)
fns.sort()

for fn in fns:
    with open(os.path.join(json_path, fn), "r") as f:
        data = json.load(f)
    data["imagePath"] = path_win2lin(data["imagePath"])
    if not path_valid(data["imagePath"]):
        image_path = path_correction(data["imagePath"])
        if path_valid(image_path):
            print(f"{fn}: {data['imagePath']} -> {image_path}")
            data["imagePath"] = image_path
        else:
            print(f"Failed to correct path: {data['imagePath']} in file {fn}")
    with open(os.path.join(json_path, fn), "w") as f:
        json.dump(data, f, indent=4)
    

