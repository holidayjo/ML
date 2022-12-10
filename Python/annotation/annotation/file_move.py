import os
import shutil

path    = r'D:\OneDrive - Texas Tech University\In Texas\TTU\Laboratary\MOT\video\Eljiro\stair_frames\2\up'
entries = os.listdir(r'D:\OneDrive - Texas Tech University\In Texas\TTU\Laboratary\MOT\video\Eljiro\stair_frames\2\up')
num = 0
for entry in entries:
    print(entry,":", num)
    print(os.path.join(path, entry))
    if num % 2 == 0:
        shutil.move(os.path.join(path, entry), os.path.join(path, "half", entry))
    num += 1


    


