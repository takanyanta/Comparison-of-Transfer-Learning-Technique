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

|VGG16|ResNet18|
---|---
|![Extract the frame](https://github.com/takanyanta/Comparison-of-Transfer-Learning-Technique/blob/main/VGG16.png "process1")|![Extract the frame](https://github.com/takanyanta/Comparison-of-Transfer-Learning-Technique/blob/main/ResNet18.png "process1")|

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
### 3-5. Training and Save the Test Results

* For training, define {batch_size, epochs} as {4, 50}
* Training is done 5 times seperately.
* Finally, Summarize the **accuracy_score** and **confusion_matrix**

```python
def save_result(array1, array2):

    n1 = np.array( [np.array([i, np.nan, np.nan]) for i in array1] ).ravel()
    n1 = n1.reshape(15, 1)
    n2 = np.concatenate( (array2[0],array2[1],array2[2], array2[3], array2[4]))
    n3 = np.hstack([n2, n1])
    df = pd.DataFrame(n3)
    return df

vgg16_tfl_nodataaugument_acc = []
vgg16_tfl_nodataaugument_cm = []
vgg16_tfl_nodataaugument_model = []
for i in range(5):
    model = training_model(transfer_learning_model(model_vgg16))
    #X_train_augumented, y_train_augumented = data_augumentation(X_train, y_train)
    model.fit(X_train, y_train, batch_size=4, epochs=50)#, callbacks=[es])

    y_pred = np.array([np.argmax( model.predict(X_test[i:i+1]) ) for i in range(len(X_test))])
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_pred, y_true)

    cm = confusion_matrix(
        np.array([np.argmax( model.predict(X_test[i:i+1]) ) for i in range(len(X_test))]),
        np.argmax(y_test, axis=1)
    )
    vgg16_tfl_nodataaugument_acc.append(acc)
    vgg16_tfl_nodataaugument_cm.append(cm)
    #vgg16_tfl_nodataaugument_model.append(model)
```

## 4. Results

* Be careful that **accuracy_score** and **confusion_matrix** are averaged.
* Class1:C3PO, Class2:Chewbacca, Class3:R2D2

### 4-1. Case 1(VGG16, Transfer Learning, non-Data Augmentated)

* Accuracy:**0.759322033898305**

|-|Class1|Class2|Class3|
---|---|---|---
|**Class1**|13.0|1.6|2.4|
|**Class2**|7.4|18.6|1.4|
|**Class3**|0.6|0.8|13.2|


### 4-2. Case 2(VGG16, Transfer Learning, Data Augmentated)

* Accuracy:**0.776271186440678**

|-|Class1|Class2|Class3|
---|---|---|---
|**Class1**|14.2|1.8|3.2|
|**Class2**|4.6|18.0|0.2|
|**Class3**|2.2|1.2|13.6|

### 4-3. Case 3(VGG16, Fine Tuning, non-Data Augmentated)

* Accuracy:**0.39661016949152544**

|-|Class1|Class2|Class3|
---|---|---|---
|**Class1**|7.4|5.0|4.2|
|**Class2**|13.6|18.0|0.2|
|**Class3**|2.2|1.2|13.6|

### 4-4. Case 4(VGG16, Fine Tuning, Data Augmentated)

* Accuracy:**0.4610169491525424**

|-|Class1|Class2|Class3|
---|---|---|---
|**Class1**|8.8|5.6|4.6|
|**Class2**|11.4|15.4|9.4|
|**Class3**|0.8|0.0|3.0|

### 4-5. Case 5(ResNet18, Transfer Learning, non-Data Augmentated)

* Accuracy:**0.7118644067796611**

|-|Class1|Class2|Class3|
---|---|---|---
|**Class1**|13.4|3.0|4.4|
|**Class2**|6.8|17.6|1.6|
|**Class3**|0.8|0.4|11.0|

### 4-6. Case 6(ResNet18, Transfer Learning, Data Augmentated)

* Accuracy:**0.7084745762711864**

|-|Class1|Class2|Class3|
---|---|---|---
|**Class1**|12.6|2.4|4.0|
|**Class2**|5.4|17.2|1.0|
|**Class3**|3.0|1.4|12.0|

### 4-7. Case 7(ResNet18, Fine Tuning, non-Data Augmentated)

* Accuracy:**0.7152542372881354**

|-|Class1|Class2|Class3|
---|---|---|---
|**Class1**|12.6|4.4|2.8|
|**Class2**|4.6|15.8|0.4|
|**Class3**|3.8|0.8|13.8|

### 4-8. Case 8(ResNet18, Fine Tuning, Data Augmentated)

* Accuracy:**0.7830508474576272**

|-|Class1|Class2|Class3|
---|---|---|---
|**Class1**|14.6|2.4|2.8|
|**Class2**|3.8|17.6|0.2|
|**Class3**|2.6|1.0|14.0|

## Conclusion

* For VGG16, Data Augmentatation improves its accuracy
* For VGG16, Fine Tuning has a bad effect on its accuracy.(Is this because of local optima?)
* For ResNet18, Fine Tuning has a good effect on its accuracy.
