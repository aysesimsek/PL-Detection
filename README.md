# **Programming Language Detection from Images**

This study has been published in 2020. Please cite this paper in your publications if it helps your research:
   
   - Comparison of Image‑Based and Text‑Based Source Code Classifcation Using Deep Learning [[**paper**]](https://github.com/aysesimsek/PL-Detection/blob/main/Comparison%20of%20Image‑Based%20and%20Text‑Based%20Source%20Code%20Classifcation%20Using%20Deep%20Learning.pdf) (**Springer 2020**)


    Kiyak, E.O., Cengiz, A.B., Birant, K.U. et al. Comparison of Image-Based and Text-Based Source Code Classification Using Deep Learning. SN COMPUT. SCI. 1, 266 (2020). https://doi.org/10.1007/s42979-020-00281-1

## **Getting Started**

In this study, three different datasets of source code images were classified with Convolutional Neural Network (CNN) which is a deep learning algorithm and the results were comparatively examined.


CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage.

## **System Architecture**

This project was developed with Python. Pytorch that is the open source machine learning library for python was used. PyTorch provides Tensor computation (like [NumPy](https://en.wikipedia.org/wiki/NumPy "NumPy")) with strong [GPU acceleration](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units "General-purpose computing on graphics processing units"). Fast.ai library was used to simplify training neural networks. Pip is a package-management system used to install and manage software packages written in Python and pip was used for this project. CUDA Toolkit that provides a development environment for creating high performance GPU-accelerated applications was used. jupyter notebook was used as program development environment.

In short, the system requirements are as follows;


|<p>- Python – 3.6 </p><p>- Pytorch – 1.1.0</p><p>- Fast.ai – 1.0.52</p>|<p>- Pip – 19.1</p><p>- CUDA – 9.0 </p><p>- Jupyter Notebook</p>|
| :- | :- |

## **Method**

In this study, two approaches have been tried in this section, these approaches are explained step by step. As shown in the Table 1 below. Existing generic image classification model ResNet-34 is used. Data was loaded, model created and trained with Fastai library.

Table 1. Method stages

||**First Approach**|**Second Approach**|
| - | :-: | :-: |
||Find learning rate (lr)|Train final layer|
||Train final layer using lr|Train all layers with 1 epoch|
||Train all layers using lr|Find learning rate|
||Train again final layer using lr|Train all layers using lr|

 **1. First Approach**
 
 In this approach, the layers are trained according to a learning rate obtained at first. In order, the last layer, all layers and, again, the last layer was trained. the        resulting error rate and accuracy values were examined.


 - Find learning rate (lr): 

 According to the results obtained from the studies, it can be said that the best learning rate is associated with the steepest drop in loss. According to this claim, it is wise  to determine the values in the range 1e-3 and 1e-2 as optimum learning rate for the graph in Figure 1.


 |![image](https://user-images.githubusercontent.com/37701256/137492543-22f6840c-ee37-476f-8cf5-f812525008fe.png)|
 | :-: |
 Figure 1. Learning rate graph sample

 - Train final layer using lr:

 In the example in Table 2, the final layer was trained in 6 epoch and learning rate was taken as 2e-2 according to Figure 1.

 Table 2. Sample training of final layer

 |**Epoch**|**Error Rate**|**Accuracy**|
 | :-: | :-: | :-: |
 |0|0.677500|0.322500|
 |1|0.630000|0.370000|
 |2|0.170000|0.830000|
 |3|0.116.250|0.883750|
 |4|0.070000|0.930000|
 |5|0.078750|0.921250|


 - Train all layers using lr:

 To increase the accuracy value or to decrease the error rate, all layers are trained with the learning rate obtained in the first step.

 In the example in Table 3, the all layers were trained in 8 epoch and learning rate was taken as 2e-2 according to Figure 1.


 Table 3. Sample training of all layers

 |**Epoch**|**Error Rate**|**Accuracy**|
 | :-: | :-: | :-: |
 |0|0.066250|0.933750|
 |1|0.068750|0.931250|
 |2|0.060000|0.940000|
 |3|0.056250|0.943750|
 |4|0.060000|0.940000|
 |5|0.052500|0.947500|
 |6|0.055000|0.945000|
 |7|0.052500|0.947500|


 - Train again final layer using lr: 

 Unlike Table 2, in Table 3 the last layer is again trained in 5 epoch. This step was the last part of this approach. finally, the received accuracy values are evaluated.

 **2. Second Approach**
 
 Unlike the first approach, this approach primarily trains the last layer and all layers with a default lr value. As a result of this process, based on the calibrated model, lr value is found and all layers are re-trained with this lr value.  

 - Train final layer:  As in the example shown in Table 2, the final layer is trained without the lr value given.

 - Train all layers with 1 epoch: All layers are trained in 1 epoch without the lr value giving.

 - Find learning rate: 

 lr value is obtained according to the calibrated model. According to Figure 2, optimum lr is determined in the range of 2e-3, 1e-6.    


 |![image](https://user-images.githubusercontent.com/37701256/137492794-88cef33b-8a4a-4cf9-ad3f-da5520f167e7.png)|
 | :-: |
 Figure 2.  Learning rate graph sample from calibrated model

 - Train all layers using lr: According to the obtained lr value, all the layers are re-trained in 8 epoch.

## **Dataset Description**

3 different data sets were used in this project. The batch size is set to the same and 64 for all data sets. Default values are used to train and validate datasets (train = 0.8, validation = 0.2). 

1. Dataset-1:

This data set contains 8 different programming language folders.


|<p>- C</p><p>- C++</p><p>- C#</p><p>- Go</p>|<p>- Python</p><p>- Ruby</p><p>- Rust</p><p>- Java</p>|
| - | - |

This data set was obtained from the codes in the master branches of the most popular titles in these 8 languages in GitHub. The codes are downloaded from master branches and 1Mb of source files each was manually selected then combined to one bundle per programming language. 17-line samples from each of the bundles are rendered into simple white-text-on-black-background images. The image sizes are 299x299. For each programming languages there is 500 images. There is a total 500\*8=4.000 data. 

2. Dataset-2:

This data set contains 6 different programming language folders.


|<p>- C</p><p>- C++</p><p>- C#</p>|<p>- Go</p><p>- Java</p><p>- Python</p>|
| - | - |

This is a subset of the Zenodo-ML Dinosaur Dataset [Github] that has been converted to small png files and organized in folders by the language. The images are in the form of white code on black background. The image sizes are 80x80. For each programming languages there is 500 images. There is a total 500\*6=3.000 data. 

3. Dataset-3:

This data set contains 6 different programming language folders.


|<p>- C</p><p>- C++</p><p>- C#</p>|<p>- Go</p><p>- Java</p><p>- Python</p>|
| - | - |

This data set was obtained from the same source as the data set 2. that is, this data set consists of images with the same characteristics. Unlike the Dataset-2, there are 10,000 images for each programming language. So there is a total 10.000\*6=60.000 data.

## **Experimental Results and Conclusion**

1. Hyperparameters of 1st Approach: 


|<p>- Final layer:</p><p>Learning rate: 2e-2</p><p>Epoch: 6</p>|<p>- All layer:</p><p>Learning rate: slice(1e-5, 1e-4)</p><p>Epoch: 8</p><p></p>|<p>- Final layer:</p><p>Learnin rate: 1e-3</p><p>Epoch: 5</p><p></p>|
| :- | :- | :- |

2. Hyperparameters of 2nd Approach:


|<p>- Final layer:</p><p>Learnin rate: Default</p><p>Epoch: 6</p><p></p>|<p>- Train all once:</p><p>Learning rate: Default</p><p>Epoch: 1</p><p></p>|<p>- All layer:</p><p>Learning rate: slice(X, Y)</p><p>Epoch: 8</p><p></p>|
| :- | :- | :- |


The accuracy values obtained by following the steps in 1st approach and 2nd approach from the datasets according to the above given parameters are shown in Table 3 and Table 4. It is observed that the model with the learning rate obtained from the calibrated model gives better accuracy than 1st approach as it is done in 2nd approach.

Table 4. Experimental results for 1st approach	

|**Dataset**|**Lr**|**Accuracy from Training of Final Layer**|**Lr (Slice)**|**Accuracy from Training of All Layers**|**Lr**|**Accuracy from Training of Final Layer**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|Dataset-1|2e-2|.91|(1e-5, 1e-4)|.93|1e-3|.92|
|Dataset-2|2e-2|.92|(1e-5, 1e-4)|.93|1e-3|.94|
|Dataset-3|2e-2|.91|(1e-5, 1e-4)|.93|1e-3|.92|



Table 5. Experimental results for 2nd approach

|**Dataset**|**Accuracy from Training of Final Layer**|**Accuracy from Training of All Layers**|**Lr**|**Accuracy from Training of Final Layer**|
| :-: | :-: | :-: | :-: | :-: |
|Dataset-1|.67|.83|2e-3, 1e-6|.96|
|Dataset-2|.68|.83|2e-3, 1e-6|.95|
|Dataset-3|.85|.93|1e-4, 1e-6|.95|

Accuracy variation according to epoch number was also observed. The accuracy value fluctuating up to a certain number of epoch, after a point, has started to be fixed.

 |![image](https://user-images.githubusercontent.com/37701256/137492907-02df551b-87a4-4b98-b515-f4ba0a521bfc.png)|
 | :-: |
 Figure 3. Accuracy change according to epoch number

