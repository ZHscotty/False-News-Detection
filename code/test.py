import pickle
import numpy as np

str = '🌴'
str2 = '🎧'
with open('../result/text_seg.pkl', 'rb') as f:
    text_seg = pickle.load(f)
with open('../result/label.pkl', 'rb') as f:
    label = pickle.load(f)
label = list(np.argmax(label, axis=1))
text_seg = text_seg[0]
print(len(text_seg))
print(len(label))
words = text_seg[1]

for x in range(len(text_seg)):
    print('label:', label[x], 'text:', text_seg[x])
