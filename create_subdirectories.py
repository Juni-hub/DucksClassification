import numpy as np;
import os;
import csv;

a = np.load('data/class_names.npy', allow_pickle=True)
print('a', a)
a_swap = {v: k for k, v in a[()].items()}
print('aSWAP', a_swap)
for key in a_swap:
    os.mkdir('./train_images/' + str(key))

n = 1
with open('train_images.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        folder = row['label']
        os.rename("./train_images" + row['image_path'], "./train_images/" + str(folder) + '/' + str(n) + '.jpg')
        n += 1
    

# print(a[()]['197.Marsh_Wren']);
# print(a_swap[155])
