import os
import cv2
import pandas as pd
import numpy as np
def onehotc(masks):
    code_mask = []
    for mask in masks:
        nw = list(mask.shape)[:2]
        nw.append(2)
        # print(tuple(nw))
        code = np.zeros(tuple(nw))
        code[:,:,0] = np.array(mask==0, dtype= np.int8)
        code[:,:,1] = np.array(mask==1, dtype= np.int8)

        code_mask.append(code)
    code_mask = np.array(code_mask, dtype= np.int8)
    print(code_mask.shape)
    return code_mask


def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

class loadDataset:
    def __init__(self, csv_file) -> None:
        self.data = pd.read_csv(csv_file)
        print(f"len : {len(self.data)}")
    def getDataset(self, k = 1.0):
        a = []
        b = []
        mx = len(self.data)
        for idx in range(int(mx*k)):
            print(f"Load {idx+1}/{mx} : {((idx+1)/mx)*100:3.2f}%", end="\r" if mx-1 != idx else "\n")
            img_path = self.data.iloc[idx, 1]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask_rle = self.data.iloc[idx, 2]
            mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
            a.append(cv2.resize(np.array(image), (224,224)))
            # a.append(np.array(image))
            b.append(np.round(cv2.resize(np.array(mask), (224,224))))
            # b.append(np.array(mask)[::2,::2])
            # b.append(np.array(mask, dtype=np.int8))
        print("finish")
        
        return np.array(a), np.array(b)
model.predict(Xtran[:1])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
# model.compile(optimizer=tf.optimizers.Adam(), loss=DiceLoss(), metrics=['accuracy'])

print(Xtran.shape, Ytran.shape)
print(Xtest.shape, Ytest.shape)
with tf.device("GPU:0"):
    for i in range(20):
        history = model.fit(Xtran, Ytran, batch_size=5, epochs=100, validation_data=(Xtest,Ytest))
        model.save(f"20230703_ver{i}.h5")

# RLE 인코딩 함수
import csv
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class ScoreData:
    def __init__(self, csv_file) -> None:
        self.data = pd.read_csv(csv_file)
        print(f"len : {len(self.data)}")
        
    def getData(self, model, k = 1.0):
        with open('cv.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            data = ['img_id','mask_rle']
            writer.writerow(data)
            a = []
            mx = len(self.data)
            for idx in range(int(mx*k)):
                a=[]
                print(f"Load {idx+1}/{mx} : {((idx+1)/mx)*100:3.2f}%", end="\r" if mx-1 != idx else "\n")
                img_path = self.data.iloc[idx, 1]
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                a.append(self.data.iloc[idx, 0])
                b = rle_encode(np.argmax(model.predict(np.array([image]), verbose=0)[0], axis=-1))
                a.append(b if b != '' else '-1')
                writer.writerow(a)

        print("finish")

kl = ScoreData("./test.csv")
test = kl.getData(model= model, k = 1.0)
