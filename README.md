# MTCNN

`pytorch` implementation of **inference stage** of face detection algorithm described in  


## Example
![example of a face detection](images/example.png)

## How to use it
Just download the repository and then do this
* If you want to creat your own picture dataset, then run `mtcnncap.py`. It will ask your name and will capture your photos and will save the photos in a folder(Folder name will be your given name).    
* The folder will be created in the images folder. You have to move it to the train folder and also some photos <not more than 14> in the val folder, name same as the folder in the trai folder.
* Then you have to run `all.py` which will create a `.npz` file in the main directory.
* After that run `embd.py` will also create a `.npz` file, containing the embeddings.
* Then you can run `realtime-faceet.py` to recognize in real time or can run `image-facenet.py` to recognize from an input picture.


## Other feaatures

* You can run `real_time.py` to only detect faces with MTCNN.
* If you have severel embeddings files ( `.npz` ) and you want to combine them together then you can run `com-npz.py` to combine then into one single `.npz` file. You just have to edit the code a little because the code is witten for combining 3 different `.npz` file.
* 

## Requirements
* pytorch 0.2
* tensorflow
* Pillow, numpy
* scikit-learn
* MTCNN, keras_facenet
* opencv

## Credit
This implementation is heavily inspired by:
* [pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
* (https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)  
