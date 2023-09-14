import cv2

new_img_path = '/home/dinusha/yolov7/data/test_imgs/test_0.jpg'
img = cv2.imread(new_img_path)
h, w, c = img.shape

with open('/home/dinusha/yolov7/data/test_imgs/test_0.txt') as f:
    content = f.readlines()
    for line in content:
        linep = line.split(' ')
        line = line.split('//')[0].replace(' ', '_').split('_')[1:]
        print(line)
        x1 = int((float(line[0]) - (float(line[2]) / 2)) * w)
        y1 = int((float(line[1]) - (float(line[3]) / 2)) * h)
        x2 = int((float(line[0]) + (float(line[2]) / 2)) * w)
        y2 = int((float(line[1]) + (float(line[3]) / 2)) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    out_path = 'test.jpg'
    cv2.imwrite(out_path, img)