# App-Based-COVID19-Detection-Using-CT-and-XRay-Images


![corona](https://user-images.githubusercontent.com/72692826/183350272-b6b79413-6bce-48af-9395-fc506b05bd99.gif)

## Project description
The first step in the treatment of COVID-19 is to screen patients in primary health centers or hospitals. Although the final diagnosis still relies mainly on transcription-polymerase chain reaction (PCR) tests, in the case of people with strong respiratory symptoms, the election protocol nowadays in hospitals relays on medical imaging, as it is simple and fast, thus helping doctors to identify diseases and their effects more quickly. Following this protocol, patients that are suspected of suffering COVID-19 first undergo an X-Ray session, and then in the case that more details are needed, they take a CT-scan session. As a result of this protocol, computed tomography scans (CT scans) and X-ray images are widely used in the clinic as alternative diagnostic tools for detecting COVID-19. Our goal was to implement a diagnosis system that classifies X-ray and CT images of patients' lungs into two categories: Covid and Non-Covid classes. Finally, we managed to reach high-precision models for both datasets, and you can see accuracies for each model below:

```
                  VGG16     VGG19     InceptionV3    ResNet50
 CT Dataset        92%      88%           87%           -
 CXR Dataset       95%      93%            -           85% 

```


# About the datasets

We used two datasets for this project.
  * For implementing the CT-based model, we used the [SARS-COV-2](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) ct dataset containing 1252 CT scans that are positive for SARS-CoV-2 infection (COVID-19) and 1230 CT scans for patients non-infected by SARS-CoV-2, 2482 CT scans in total.
  * For implementing an X-ray-based model, we used the [COVID-19 Radiography](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) dataset.
 * Note that due to computational limitations, we only used the first 1200 images of each class from the previously mentioned CXR dataset.


## Methodology
### CNN
The CNN-based deep neural system is widely used in the medical classification task. [CNN](https://towardsdatascience.com/a-gentle-introduction-to-neural-networks-series-part-1-2b90b87795bc) is an excellent feature extractor; Therefore, utilizing it to classify medical images can avoid complicated and expensive feature engineering. 
But instead of building a CNN model from scratch, We used pre-trained models and [transfer learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/) to train our model. 
### Which models did we use?
We used 3 pre-trained models for both datasets tp trian our model
1. [VGG16](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)

[<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/182526626-ae106b18-f71b-4a56-b61f-8fc78d0205f4.png">](https://www.researchgate.net/publication/321829624/figure/fig2/AS:571845657481217@1513350037610/VGG16-architecture-16.png)

2. [VGG19](https://iq.opengenus.org/vgg19-architecture/#:~:text=VGG19%20is%20a%20variant%20of,VGG19%20has%2019.6%20billion%20FLOPs.)

[<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/182526495-774a3816-36a6-4f9b-94e9-838f760782a0.png">](https://www.researchgate.net/figure/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means_fig2_325137356)

3. [InceptionV3](https://medium.com/@AnasBrital98/inception-v3-cnn-architecture-explained-691cfb7bba08)

[<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/182527742-c723f4d4-d322-4203-bbd8-8de0b585a77a.png">](https://production-media.paperswithcode.com/methods/inceptionv3onc--oview_vjAbOfw.png)

4. [Resnet50](https://cv-tricks.com/keras/understand-implement-resnets/)

[<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/182528482-18a8f8c7-0888-40b1-83e5-51bc641a833c.png">](https://www.researchgate.net/figure/The-architecture-of-ResNet-50-model_fig4_349717475)



Basically our job to make a prediction model is done in these four steps:
- Loading and Visualizing the Dataset
- Data Pre-Processing
- Building & Training our Model Using Pre-trained Models (fine-tuning is done in 20 epochs with a batch size of 32
- Save and Evaluate the model

## Results
### Metrics
- Precision

The precision is the ratio tp / (tp + fp), where tp is the number of true positives and fp is the number of false positives. The precision is intuitively the ability of the classifier not to label a negative sample as positive.

<img width="300" alt="image" src="https://user-images.githubusercontent.com/72692826/184919263-9b9718a0-06e5-4db3-a302-5a44b6c61b41.png">

- F1-score

The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. It is primarily used to compare the performance of two classifiers.

<img width="300" alt="image" src="https://user-images.githubusercontent.com/72692826/184919662-966c7a26-4435-4c16-a76f-14a5243269eb.png">

- Recall

The recall is the ratio tp / (tp + fn), where tp is the number of true positives and fn is the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

<img width="300" alt="image" src="https://user-images.githubusercontent.com/72692826/184919477-671c9798-2fb6-4669-9800-b22241a9bca0.png">

- Support

The support is the number of occurrences of each class in y_true.

### Results of CT Dataset 
- VGG16

<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/183387982-192dc048-9d68-4517-b8ea-c4ccb96a1df7.png">

<img width="390" alt="image" src="https://user-images.githubusercontent.com/72692826/183388054-b1a3e9ff-c3fb-4ae1-89ec-19e28b16d8cb.png">

- VGG19

<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/183388221-832b8861-454d-4281-9b3e-0a4cd304f702.png">

<img width="390" alt="image" src="https://user-images.githubusercontent.com/72692826/183388263-46aa2e2b-87f4-406e-a697-bd89f2615174.png">

- InceptionV3

<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/183388522-2a2d8b65-8473-46c6-bf3b-43573a5a8c48.png">

<img width="390" alt="image" src="https://user-images.githubusercontent.com/72692826/183388570-e3a2f4fc-0bfd-43cc-86cd-4d08de5c052c.png">

### Results of CXR Dataset 
- VGG16

<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/183388678-79c9d536-5103-407a-a5a6-fcd999599acc.png">

<img width="390" alt="image" src="https://user-images.githubusercontent.com/72692826/183388713-499dbc94-e6a6-44f9-9e4b-e90f858c3545.png">

- VGG19

<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/183388860-6c5c6f72-9223-47e0-b593-6af2ebdbbbf2.png">

<img width="390" alt="image" src="https://user-images.githubusercontent.com/72692826/183388896-d5adb5f7-d0aa-44e6-a70c-b41c8447a772.png">

- ResNet50

<img width="450" alt="image" src="https://user-images.githubusercontent.com/72692826/183943004-ad73aba4-1b80-4e22-90f2-db367fa281f4.png">


<img width="390" alt="image" src="https://user-images.githubusercontent.com/72692826/183943072-240ce606-2963-4709-b93a-d93ec01d7147.png">


## Flask application
We've developed a flask application for you to use our models easily. We've tried to create a simple and beautiful UI so that you can easily find what you want.

To use our models, select your CT-Scan/X-Ray image and model on the detect page of the application.

### How to run locally
1. Download this repository

2. Open "libraries.txt" and install the necessary libraries (if you don't have them)

3. Download our models for [CT-Scan](https://drive.google.com/drive/folders/1L9NXbpr_ya6T29J6YGvIyVIl15ZZwBEn) and [X-Ray](https://drive.google.com/drive/folders/1MJuHUZGhgL6rv8_TViR3XKbrm2BZ4bmW)

4. Copy model files (like ResNet50_CXR_model.h5) and paste them into the proper folder in the project folder/model
For example, you should copy ResNet50_CXR_model.h5 and paste it into model/resnet50 in the project folder

5. Run the following command in the project folder using terminal
```
flask run
```

6. Paste http://127.0.0.1:5000 in your browser
