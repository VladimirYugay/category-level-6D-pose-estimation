# Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation


## Basic definitions


Before we go to the content of the paper itself, there might be some definitions and principles that might be useful when reading the article:

* [6D pose](https://en.wikipedia.org/wiki/Pose_(computer_vision))
* [Depth image](https://en.wikipedia.org/wiki/Depth_map)
* [CAD model](https://de.wikipedia.org/wiki/CAD)

## Introduction

Estimating 6D pose of objects has been always an interesting and challenging task for many applications in such spheres as robotics, augmented reality and 3D scene understanding. For instance, when a robot wants to grasp a box, it's necessary for him to know what rotation and translation does it have w.r.t. to him in order to adjust his hand properly. Another example is when we want to understand what's going on on the image containing a group of people. If we have estimates of theirs 6D poses we can make a better guess what exactly are the doing.

In this article we'll have a look at one possible approach to estimate 6D poses of objects from particular categories (category-level) using only a 3D image.


## Related Work

The authors of the paper propose not only an effective model for estimating the pose, but also a new way to construct a training dataset. However, in order to understand what's special and new in their approach we need to have a quick look at related problems and their solutions.

**Category-Level 3D Object Detection.** In this problem the main objective is to estimate the bounding box around objects from certain categories using a 3D image. There are two main branches how people solved this problem:

* Directly work with volumetric data [1, 2]. The high-level idea behind this approach is to convert the whole CAD model to the 3D point cloud and work with some regions of it to predict the coordinates of the bounding box and the object class which a particular region contains
* Use 2D object proposals to infer the bounding box [3, 4, 5, 6]. The intuition behind this approach is to construct 3D [image segmentation](https://blog.playment.io/semantic-segmentation/) out of 2D segmented images. Since for 2D there exist lots of models which already give good results they're used to extend it to a much more difficult to segment directly 3D space

***Difference.*** As you might have already seen, the problem of estimating 3D bounding box is just a sub-problem which the authors have solved (to a certain extent) in this work.

**Instance-Level 6D Pose Estimation.** This problem statement means that we're are not bounded to any categories. We predict 6D pose of *all* the objects no matter to which class they belong which are present on a given image. The two main approaches to solve this problem are:

* Template matching approach [7, 8, 9, 10]. The intuition is to get the 3D point clouds of the objects in the image and after that to match CAD models of the corresponding objects to them. After matching, it's easy to see what translation and rotation were applied to a given CAD model to be matched to the 3D point cloud
* Coordinates regression [11, 12, 13]. In this branch, the idea is to directly predict the object surface position corresponding to each object pixel

***Difference.*** Both of the above-mentioned approaches require CAD models for all unseen objects during test and train time. Storing the models for all unseen objects during test time is infeasible.

**Category-Level 4D Pose Estimation.** For this problem, only translation and rotation along a normal axis are predicted. For example, in [4] 4D poses of indoor furniture were estimated. It's clear that a chair or a table can't be on the ceiling and that they always stand on the floor. Consequently, predicting rotation along the floor is enough to solve the task of pose estimation.

***Difference.*** While for estimating pose of indoor furniture 4D pose estimation is enough, in real life there are many cases when objects can be anywhere in space and there is no normal axis. For such cases, 4D pose estimation is not enough.

**Training Data Generation.** Generating data for pose estimation is a quite tedious and expensive task. There are two major approaches to solve it:

* Manual data generation [14, 15, 16]. There's a series of works where researchers tried to generate and label data manually, but the datasets are relatively small
* Synthetic data generation [17, 18, 19, 20]. Several attempts were made to generate synthetic scenes and put synthetic objects on them. However, for the sake of simplicity the scenes were lacking important details such as color, texture and proper objects distribution

***Difference.*** In this approach, the authors propose a method for generating synthetic training data   considering real scenes and inserting synthetic objects in a more "realistic" places on the image. Moreover, the authors are focusing more on classes of smaller objects like mugs or laptops.

##Method

As summary, we want to estimate 6D pose of objects from a fixed set of categories and we have no CAD models for unseen objects during test time. For unseen objects 6D pose is not well-defined and to add we need some way to represent it.

###Normalized Object Coordinate Space (NOCS)
NOCS is a 3D space contained withing a unit cube (each coordinate is between zero and one). For each of categories a separate NOCS space is defined. CAD model of each object of each categories (remember, we don't have CAD models only for unseen objects during test time!) is normalized such that its diagonal is equal to one and all the objects withing each categories are rotated in a same way.

This is how an object from camera category looks like in NOCS space (here and further we'll set axis in NOCS space to represent RGB values):

<br/>
<br/>
<center>
![Alt text](NOCS.png)
</center>
<div align="center" style="font-size:12px">Fig. 1, He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, Leonidas J. Guibas,Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation, CVPR 19</div>
<br/>


###Network Architecture
The pose is estimated with the help of a neural network. However, in order to better understand how exactly it works, it's better to have a quick look at works on which this architecture is based.

**Faster RCNN.** We'll skip the first two works of RCNN series and start considering Faster-RCNN [21] since it's more relevant for the method. Faster-RCNN is a network for object detection, which predicts the frame where the object is contained and class probabilities about what object is in the frame. The network extracts features from the whole image once, then a separate sub-network proposes regions where the object might be, features from these particular regions are further passed to another sub-network from which two heads output bounding box coordinates and class probabilities about which object is in the box. The cross-entropy loss is used for predicting class probabilities and soft L1 loss is used for box coordinates regression. Soft L1 loss is used to prevent gradient explosion.

<br/>
<br/>
<center>
![Alt text](faster-rcnn.png)
</center>
<div align="center" style="font-size:12px">Fig. 2, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun,  arXiv:1506.01497</div>
<br/>


**Mask RCNN.** An extension of Faster-RCNN is Mask-it it's computedRCNN [22]. Besides predicting class probabilities of the frame and coordinates of the box containing the object, Mask RCNN also predicts the mask of the object i.e. each object is colored with a color corresponding to his class. In terms of architecture, the only difference from Faster RCNN is that an additional head predicting masks is added. Cross-entropy loss is used for mask prediction since the problem of mask prediction is considered as pixel-wise classification problem.

<br/>
<br/>
<center>
![Alt text](mask-rcnn.png)
</center>
<div align="center" style="font-size:12px">Fig. 3, Mask R-CNN, Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick,  arXiv:1703.06870</div>
<br/>


**Proposed Architecture.** The architecture proposed for 6D pose estimation is based on Mask-RCNN. First, Additional head is added for predicting the NOCS coordinates of the object. After both segmentation mask and NOCS mask are obtained, two point clouds are constructed with the help of depth image. There's a nice [article](https://elcharolin.wordpress.com/2017/09/06/transforming-a-depth-map-into-a-3d-point-cloud/) explaining one way to convert 2D image and a depth image to the point cloud. After that translation and rotation computed such that after applying them to the NOCS point cloud it would be similar to the point segmentation mask point cloud in terms of euclidean distance. Both translation and rotations are estimated using Umeyama [23] algorithm.

<br/>
<br/>
<center>
![Alt text](nocs-nn.png)
</center>
<div align="center" style="font-size:12px">Fig. 4, He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, Leonidas J. Guibas,Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation, CVPR</div>
<br/>
