import cnn_model
# import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils as np_utils

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#入力と出力を指定
im_rows=32
im_cols=32
im_color=3
in_shape=(im_rows,im_cols,im_color)
nb_classes=3

#写真データを読み込み
photos=np.load("SuperviseData/FLRs.npz")
x=photos["x"]
y=photos["y"]

#読み込んだデータを三次元配列に変換
x=x.reshape(-1,im_rows,im_cols,im_color)
x=x.astype("float32")/255
#ラベルデータをone-hotベクトルに直す
y = np_utils.to_categorical(y.astype("int32"),nb_classes)
# y=keras.utils.np_utils.to_categorical(y.astype("int32"),nb_classes)
#学習用とテスト用に分ける
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
#cnnモデルを取得
model=cnn_model.get_model(in_shape,nb_classes)
print("model.summary")
model.summary()
#学習を実行
hist=model.fit(x_train,y_train,
              batch_size=32,
              epochs=20,
              verbose=1,
              validation_data=(x_test,y_test))

#モデルを評価
score=model.evaluate(x_test,y_test,verbose=1)


# model_builder = keras.applications.xception.Xception
# img_size = (32, 32)
# preprocess_input = keras.applications.xception.preprocess_input
# decode_predictions = keras.applications.xception.decode_predictions
#
# last_conv_layer_name = "sequential"
#
# # The local path to our target image
# img_path = keras.utils.get_file(
#     "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
# )
#
# display(Image(img_path))
#
#
# # Prepare image
# img_array = preprocess_input(get_img_array(img_path, size=img_size))
#
# # Make model
# model = model_builder(weights="imagenet")
#
# # Remove last layer's softmax
# model.layers[-1].activation = None
#
# # Print what the top predicted class is
# preds = model.predict(img_array)
# print("Predicted:", decode_predictions(preds, top=1)[0])
#
# # Generate class activation heatmap
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
#
# # Display heatmap
# plt.matshow(heatmap)
# plt.show()


print("正解率=",score[1],"loss=",score[0])

#学習の様子をグラフへ描写
#正解率の推移をプロット
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["train","test"],loc="upper left")
plt.show()

#ロスの推移をプロット
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Loss")
plt.legend(["train","test"],loc="upper left")
plt.show()

model.save_weights("./SuperviseData/FLRs-model-light_add.hdf5")
