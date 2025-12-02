import os

BASE = "Plant_disases_detection2"

for root, dirs, files in os.walk(BASE):
    for name in files + dirs:
        new_name = name.rstrip()
        if new_name != name:
            old_path = os.path.join(root, name)
            new_path = os.path.join(root, new_name)
            print("Renaming:", old_path)
            os.rename(old_path, new_path)

print("✅ Done fixing all file + folder names")
