import os
folder_path = "C:/Users/Lifu Wang/Desktop/GT/Project/ReverseDistillation/content/bottle/train/generated"

count = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, str(count) + ".png")
        os.rename(src, dst)
        count += 1