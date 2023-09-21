"""
This script process LaSOT dataset into a format that can be used by yolov7 dataloader.

use wget https://huggingface.co/datasets/l-lt/LaSOT/resolve/main/bottle.zip to download the dataset
"""

import os
import cv2

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def process_one_sequence(path, class_id, global_data_idx, save_folder):
  lables_file = os.path.join(path, 'groundtruth.txt')
  img_folder = os.path.join(path, 'img')

  image_list = os.listdir(img_folder)
  image_list.sort()

  if not os.path.exists(save_folder):
    os.mkdir(save_folder)

  with open(lables_file) as f:
      content = f.readlines()

  for f1, f2 in pairwise(image_list):
    # print(f1, f2)
    if f1.endswith('.jpg') and f2.endswith('.jpg'):

      img_id = int(f1.replace('.jpg', ''))
      next_img_id = int(f2.replace('.jpg', ''))

      img = cv2.imread(os.path.join(img_folder, f2))
      H, W, C = img.shape

      seq_folder = os.path.join(save_folder, str(global_data_idx))

      if not os.path.exists(seq_folder):
        os.mkdir(seq_folder)

      os.rename(os.path.join(img_folder, f1), os.path.join(seq_folder, f1))
      os.rename(os.path.join(img_folder, f2), os.path.join(seq_folder, f2))

      txt_file = os.path.join(seq_folder, f'{global_data_idx}.txt')

      # print(txt_file)
      # print(next_img_id-1)
      with open(txt_file, 'w') as textfile:
        box = content[next_img_id-1].split(',')
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        textfile.write(f"{class_id} {(x + w / 2) / W} {(y + h / 2) / H} { w / W} {h / H}\n")

      global_data_idx += 1

  return global_data_idx


def process_one_catogory(path, clas_idx, global_data_idx, dataset_folder):
  g_id = global_data_idx
  for seq in os.listdir(path):
    ret = process_one_sequence(os.path.join(path, seq), clas_idx, g_id, dataset_folder)
    g_id = ret
    print(g_id)



global_data_idx = 1
class_id = 0
catogory_path = '/content/yolov7/data/dataset/bottle'
dataset_path = '/content/yolov7/data/dataset/tmp'

process_one_catogory(catogory_path, class_id, global_data_idx, dataset_path)