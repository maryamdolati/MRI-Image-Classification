

!pip -q install tensorflow==2.15.0

import os, glob, random, warnings, zipfile, io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["TF_XLA_FLAGS"]="--tf_xla_auto_jit=0"

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

DATA_SOURCE = "upload"
BASE_DIR = "/content/data/brain-mri"

if DATA_SOURCE == "upload":
    from google.colab import files
    uploaded = files.upload()
    zip_name = next(iter(uploaded))
    os.makedirs(BASE_DIR, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(uploaded[zip_name]), 'r') as zf:
        zf.extractall(BASE_DIR)
    cand = []
    for root, dirs, files_ in os.walk(BASE_DIR):
        if {'yes','no'}.issubset(set([d.lower() for d in dirs])):
            cand.append(root)
    if len(cand) >= 1:
        BASE_DIR = cand[0]
elif DATA_SOURCE == "drive":
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    BASE_DIR = "/content/drive/MyDrive/brain-mri-images-for-brain-tumor-detection"

IMG_SIZE = 128
def load_folder_images(folder, label):
    data = []
    files = glob.glob(os.path.join(folder, "*"))
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append((img, label))
    return data

path_yes = os.path.join(BASE_DIR, "yes")
path_no  = os.path.join(BASE_DIR, "no")

tumor = load_folder_images(path_yes, 1)
no_tumor = load_folder_images(path_no, 0)

all_data = tumor + no_tumor
all_data = shuffle(all_data, random_state=42)
data = np.array([x[0] for x in all_data], dtype=np.float32)
labels = np.array([x[1] for x in all_data], dtype=np.int32)

plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(2,3,i+1); plt.imshow(tumor[i][0]); plt.title("Tumor: Yes"); plt.axis('off')
for i in range(3):
    plt.subplot(2,3,i+4); plt.imshow(no_tumor[i][0]); plt.title("Tumor: No"); plt.axis('off')
plt.tight_layout(); plt.show()

u, c = np.unique(labels, return_counts=True)
plt.bar(u, c); plt.xticks([0,1], ['No Tumor','Tumor']); plt.title('Class Distribution'); plt.show()

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42, stratify=labels)
x_train = x_train/255.0
x_test  = x_test/255.0

cls_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y_train)
class_weights = {0: cls_weights[0], 1: cls_weights[1]}

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,3)),
    MaxPooling2D((2,2)),
    BatchNormalization(),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),
    Flatten(),
    Dropout(0.5),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class StopAt99(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy',0) > 0.99:
            self.model.stop_training = True

callbacks = [
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    ModelCheckpoint('brain_tumor_best.h5', monitor='val_loss', save_best_only=True),
    StopAt99()
]

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss'); plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

model.save('brain_tumor.h5')
