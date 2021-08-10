

import os
import pickle
from PIL import Image
import csv


path = './'
save = '../'
file_list = ['mini-imagenet-cache-train.pkl', 'mini-imagenet-cache-val.pkl', 'mini-imagenet-cache-test.pkl']
mode_list = ['train', 'val', 'test']
for file_name, mode in zip(file_list, mode_list):
    data_file = path + file_name
    print(data_file)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        images, classes = data['image_data'], data['class_dict']

    t = 0
    file1 = open(f"{save}/{mode}.csv","w")#write mode
    writer = csv.writer(file1, delimiter=' ', lineterminator='\n',  escapechar=' ', quoting=csv.QUOTE_NONE)
    writer.writerow(['fname,label'])

    for name, indices in classes.items():
        data_img =images[indices]
        # print(name, data_img.shape)
        out_path = f'{save}/{mode}/{name}/'
        os.makedirs(out_path, exist_ok=True)
        for ind  in indices:
            i = int(ind%600)
            img_label = f'{name}_{i:08}.jpg,{name}'

            file = out_path + f'{name}_{i:08}.jpg'

            pil_img = Image.fromarray(data_img[i])
            print(img_label)
            writer.writerow([f'{img_label}'])
            print(file)
            # plt.imshow(pil_img)
            pil_img.save(file)
            # if i==5: break
        
        t +=1
        # if t==5: break
    file1.close()