# Comparison-of-Transfer-Learning-Technique
Convolutional Neural Networks, Transfer Learning, Fine Tuning, Data Augmentation, VGG-16, ResNet-18, Classification, Keras

## 1. Objective

* To try to implement CNN code, which uses pretrained networks.
* To compare 2 pretrained networks by changing conditions.
* Considering cases are below;

|Case|Network|Learning Paradigm|Data Augmentation|
---|---|---|---
|1|VGG16|Transfer Learning|Do|
|2|//|//|Not Do|
|3|//|Fine Tuning|Do|
|4|//|//|Not Do|
|5|ResNet18|Transfer Learning|Do|
|6|//|//|Not Do|
|7|//|Fine Tuning|Do|
|8|//|//|Not Do|

## 2. Theme for Classification

* Define the theme as the classification task of characters which appears in Star Wars movie.
* Actually, choose "C3PO", "R2D2", and "Chewbacca".
* Those pictures are searched on the net.

|C3PO|R2D2|Chewbacca|
---|---|---
|![Extract the frame](https://github.com/takanyanta/Comparison-of-Transfer-Learning-Technique/blob/main/sample_pic/C3Po.png "process1")|![Extract the frame](https://github.com/takanyanta/Comparison-of-Transfer-Learning-Technique/blob/main/sample_pic/R2D2.jpg "process1")|![Extract the frame](https://github.com/takanyanta/Comparison-of-Transfer-Learning-Technique/blob/main/sample_pic/chewbacca.jpg "process1")

## 3. Implementation

### 3-1. Load Networks

#### 3-1-1. VGG16

```python
from tensorflow.keras.applications import vgg16
model_vgg16=vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model_vgg16.summary()

plot_model(model_vgg16)
```

#### 3-1-2. ResNet18

* Use [image-classifiers](https://github.com/qubvel/classification_models) to get pretrained model
* The model above is composed in the way of **Keras** manner.
* For Unifing **tensorflow.keras** manner, once save the model and the weights to json file and .h5 file, then load them.

```python
model_resnet18 = model_from_json(open('model_resnet18.json', 'r').read())
model_resnet18.load_weights("model_resnet18.h5")

plot_model(model_resnet18)
```

### 3-2. Change Networks

* For transfer learning, define weights as **non-trainable**.

```python
def trainable_to_nontrainable(model):
    """
    Transfer trainable variables to non-trainable variables
    """
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    return model

def transfer_learning_model(model):
    model1 = clone_model(model)
    model1 = trainable_to_nontrainable(model1)
    return model1
```

* For fine tuning, define some layers as **trainable** and the others as **non-trainable**.

```python
def fine_tune_for_vgg16(model):
    model1 = clone_model(model)
    model1 = trainable_to_nontrainable(model1)
    model1.layers[-4].trainable=True
    model1.layers[-3].trainable=True
    model1.layers[-2].trainable=True
    return model1

def fine_tune_for_resnet18(model):
    model1 = clone_model(model)
    model1 = trainable_to_nontrainable(model1)
    model1.layers[-11].trainable=True
    model1.layers[-10].trainable=True
    model1.layers[-9].trainable=True
    model1.layers[-8].trainable=True
    model1.layers[-7].trainable=True
    model1.layers[-6].trainable=True
    model1.layers[-5].trainable=True
    model1.layers[-4].trainable=True
    model1.layers[-3].trainable=True
    model1.layers[-2].trainable=True
    model1.layers[-1].trainable=True
    return model1
```

* red marked layers are trainable

### 3-3. Prepair Datasets

* Resize images size as (224, 224, 3).
* Use almost 70 images per class.
* Define 30% of data as test data.

```python
def prepair_dataset(folder_name):
    """
    Get image fron folders
    resize image_shape to (224, 224)
    """
    master_list = []
    for i in glob.glob("./{}/*".format(folder_name)):
        pic_list = glob.glob(i + "/" + "*")
        temp_list = np.empty((len(pic_list), 224, 224, 3) ,dtype=np.uint8)
        k = 0
        for j, jdx in enumerate(pic_list):
            try:
                img = Image.open(jdx)
                img = np.array(img, dtype=np.uint8)
                img = cv2.resize(img, (224, 224))
                if img.ndim == 3 and img.shape[2] == 3:
                    temp_list[k, :, :, :] = img[:, :, :]
                    k += 1
            except UnidentifiedImageError:
                print(j)
                continue

        master_list.append(temp_list)
    return master_list

kimtesu_datasets = prepair_dataset("AnimationPicture")

y3 = np.array([np.array([[1.0, 0.0, 0.0] for i in range(len(kimtesu_datasets[0]))]).astype("float32"),
      np.array([[0.0, 1.0, 0.0] for i in range(len(kimtesu_datasets[1]))]).astype("float32"),
      np.array([[0.0, 0.0, 1.0] for i in range(len(kimtesu_datasets[2]))]).astype("float32")])

for i, idx in enumerate(kimtesu_datasets):
    if i == 0:
        X_train, X_test, y_train, y_test = train_test_split(kimtesu_datasets[i][0:70, :, :, :], y3[i][0:70, :], test_size=0.3, random_state=0)
    else:
        X_train1, X_test1, y_train1, y_test1 = train_test_split(kimtesu_datasets[i][0:70, :, :, :], y3[i][0:70, :], test_size=0.3, random_state=0)
        X_train = np.concatenate((X_train, X_train1))
        X_test = np.concatenate((X_test, X_test1))
        y_train = np.concatenate((y_train, y_train1))
        y_test = np.concatenate((y_test, y_test1))
```

* For data augumentation, inversion of images.

```python
def data_augumentation(X, y):
    augumented_data = np.empty(X.shape, dtype=np.uint8)
    for i, idx in enumerate(X):
        augumented_data[i, :, :, :] = cv2.flip(idx, 1)
    return np.concatenate((X, augumented_data)), np.concatenate((y, y))
```

### 3-4. Modify Networks

* Add 2 hidden layers and dropouts

```python
def training_model(model_):
    model = Sequential()
    model.add(model_)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation=softmax))

    model.compile(loss = categorical_crossentropy, optimizer = "adam", metrics=["accuracy"])
    #model.summary()
    return model
```
