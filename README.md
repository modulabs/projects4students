# Projects for students 
* Final update: 2019. 05. 21.
* All right reserved @ ModuLabs 2019


## Getting Started

### Prerequisites
* [`TensorFlow`](https://www.tensorflow.org) above version 1.13
* Python 3.6 (recommend Anaconda)
* Python libraries:
  * `numpy`, `matplotblib`, `pandas`
  * `PIL`, `imageio` for images
  * `fix_yahoo_finance` for stock market prediction
* Jupyter notebook
* Ubuntu, OS X and Windows OS



## CNN projects

### Task
* GIANA dataset으로 위내시경 이미지에서 용종을 segmentation 해보자.
* 데이터 불러오기를 제외한 딥러닝 트레이닝 과정을 직접 구현해보는 것이 목표 입니다.
* This code is borrowed from [TensorFlow tutorials/Image Segmentation](https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb) which is made of `tf.keras.layers` and `tf.enable_eager_execution()`.
* You can see the detail description [tutorial link](https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb)  

### Dataset
* I use below dataset instead of [carvana-image-masking-challenge dataset](https://www.kaggle.com/c/carvana-image-masking-challenge/rules) in TensorFlow Tutorials which is a kaggle competition dataset.
  * carvana-image-masking-challenge dataset: Too large dataset (14GB)
* [Gastrointestinal Image ANAlys Challenges (GIANA)](https://giana.grand-challenge.org) Dataset (345MB)
  * Train data: 300 images with RGB channels (bmp format)
  * Train lables: 300 images with 1 channels (bmp format)
  * Image size: 574 x 500
* Training시 **image size는 256**으로 resize

### Baseline code
* Dataset: train, test로 split
* Input data shape: (`batch_size`, 256, 256, 3)
* Output data shape: (`batch_size`, 256, 256, 1)
* Architecture: 
  * 간단한 Encoder-Decoder 구조
  * U-Net 구조
  * [`tf.keras.layers`](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 사용
* Training
  * `tf.data.Dataset` 사용
  * `model.fit()` 사용 for weight update
* Evaluation
  * MeanIOU: Image Segmentation에서 많이 쓰이는 evaluation measure
  * tf.version 1.13 API: [`tf.metrics.mean_iou`](https://www.tensorflow.org/api_docs/python/tf/metrics/mean_iou)
    * `tf.enable_eager_execution()`이 작동하지 않음
    * 따라서 예전 방식대로 `tf.Session()`을 이용하여 작성하거나 아래와 같이 2.0 version으로 작성하여야 함
  * tf.version 2.0 API: [`tf.keras.metrics.MeanIoU`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics/MeanIoU)

### Try some techniques
* Change model architectures (Custom model)
  * Try another models (Unet 모델)
* Various regularization methods





## Authors
* [Il Gu Yi](https://github.com/ilguyi)
