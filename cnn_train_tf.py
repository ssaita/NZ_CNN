import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input

import subprocess

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# physical_devices = tf.config.list_physical_devices('GPU')
# if not physical_devices:
#     pass
# else:
#     tf.config.set_visible_devices(physical_devices[0], 'GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# !nvidia-smi
# subprocess.run('nvdia-smi')

# モデルの作成（pretrained modelを使用する）
from tensorflow.keras.applications.resnet50 import ResNet50 as pretrained_model
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input

""" 別のpretrained modelを使用する用に 入力画像のサイズが変わったりするので注意
# VGG16 input_shape = (224,224,3)
from tensorflow.keras.applications.vgg16 import VGG16 as pretrained_model
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input

# InceptionV3 input_shape = (299,299,3)
from tensorflow.keras.applications.inception_v3 import InceptionV3 as pretrained_model
from tensorflow.keras.applications.inception_v3 import decode_predictions, preprocess_input

# EfficientNetB2  input_shape = (260,260,3)
from tensorflow.keras.applications.efficientnet import EfficientNetB2 as pretrained_model
from tensorflow.keras.applications.efficientnet import decode_predictions, preprocess_input
"""

model = pretrained_model(weights="imagenet")

# モデルの確認
model.summary()

# テストデータの用意
# !mkdir input_images
# subprocess.run('mkdir input_images')

#テストデータの読み込みと確認
img_path="SuperviseData/group1/MDMEYR_2011-08-22.png"
img_rgb = image.load_img(img_path,color_mode="rgb",target_size=(224,224))

img = image.img_to_array(img_rgb)
test_img = deepcopy(img)

plt.imshow(img / 255.)
plt.show()

# 予測結果
img = preprocess_input(img[np.newaxis,...])
preds = model.predict(img)

results = decode_predictions(preds,top=5)
[print(r) for r in results[0]]
print()

# Grad-CAM
# Reference: https://keras.io/examples/vision/grad_cam/
def make_gradcam_heatmap(img,model,target_layer_name,pred_index=None):
    grad_model = tf.keras.Model(
        [model.inputs],[model.get_layer(target_layer_name).output,model.output]
    )

    with tf.GradientTape() as tape:
        target_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:,pred_index]

    grads = tape.gradient(class_channel,target_layer_output)
    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2)) # αの計算

    heatmap = target_layer_output[0] @ pooled_grads[...,tf.newaxis] # α * A^k
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# Grad-CAMの実行
target_layer_name = "conv5_block3_out"
heatmap = make_gradcam_heatmap(img,model,target_layer_name)

# 画像と同サイズにupsampling
# order=0 でただの拡大 order=1~5 で滑らかに拡大
gradcam = resize(heatmap,(224,224),order=1,mode="reflect",anti_aliasing=False)

# Grad-CAMの結果表示
fig = plt.figure(figsize=(6,4))
fig.add_subplot(1,3,1)
plt.imshow(test_img / 255.)
plt.imshow(gradcam ,cmap="jet",alpha=0.5)
fig.add_subplot(1,3,2)
plt.imshow(test_img / 255.)
fig.add_subplot(1,3,3)
plt.imshow(heatmap,cmap="jet")
plt.show()
