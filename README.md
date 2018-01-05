# transforms

This project is used for image augmentation, featured in simultaneous transformation of image, keypoints, bounding boxes, and segmentation mask. It's extended from [torchvision](https://github.com/pytorch/vision) and project [imageUtils](https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0). Currently the implemented transformations include ColorJitter, RandomErasing, Expand, Scale, Resize, Crop, ElasticTransform, Rotate, Shift, and Flip. If you need more image augmentation types, you can take a look at [imgaug](https://github.com/aleju/imgaug), it's a very comprehensive library.

Image transformations can be divided into two categories: 
* geometric transformations
* photometric transformations

Geometric transformations alter the geometry of the image with the aim of making algorithm invariant to change in position/orientation, and to image deformation. Photometric transformations amend the color channels with the objective of making algorithm invariant to change in lighting and color.  

For computer vision problems other than image classification, transforming image alone is often not enough. Say, for object detection, we should transform image and bounding boxes simultaneously, and for image segmentation, we should also transform the segmentation mask (mask should not be seen same as image, because for image geometric transformations, we often use interpolation to make transformed images visually pleased,
but interpolation for mask is meaningless). This project concentrates on these problems.



### Keypoints
```python
PRNG = RandomState()
transform = Compose([
    [ColorJitter(), None],    # or write [ColorJitter()]
    Expand((0.8, 1.5)),
    RandomCompose([
        RandomRotate(360),
        RandomShift(0.2)]),
    Scale(512),
    RandomCrop(512),
    HorizontalFlip(),
    ], 
    PRNG, 
    border='constant', 
    fillval=0,
    outside_points='inf')

# image: np.ndarray of shape (h, w, 3), RGB format
# pts: np.ndarray of shape (N, 2), e.g. [[x1, y1], [x2, y2], ...]
transformed_image, transformed_pts = transfrom(image, pts)
```
![](https://i.loli.net/2018/01/06/5a5005a552e3b.gif)

### Bounding Boxes
bounding boxes -> vertices coordinates -> transformed coordinates -> transformed bounding boxes.  
Below is the agumentation used by [SSD](https://arxiv.org/abs/1512.02325).
```python
PRNG = RandomState()
transform = Compose([           
    [ColorJitter(prob=0.5)],
    BoxesToCoords(relative=False),
    HorizontalFlip(),
    Expand((1, 4), prob=0.5),
    ObjectRandomCrop(),
    Resize(300),
    CoordsToBoxes(relative=False),
    ], 
    PRNG, 
    mode='linear', 
    border='constant', 
    fillval=0, 
    outside_points='clamp')

# image: np.ndarray of shape (h, w, 3), RGB format
# bboxes: np.ndarray of shape (N, 4), e.g. [[xmin, ymin, xmax, ymax], ...]
# note that the bboxes can be normalized to [0, 1] (yout should set 
# relative=True accordingly) or use pixel value directly (yout should set 
# relative=False). 
transformed_image, transformed_bboxes = transfrom(image, bboxes)
```
![](https://i.loli.net/2018/01/06/5a5006787251c.gif)

### Segmentation Mask
```python
transform = Compose([
    [ColorJitter(), None],
    Merge(),
    Expand((0.7, 1.4)),
    RandomCompose([
        RandomResize(1, 1.5),
        RandomRotate(5),
        RandomShift(0.1)]),
    Scale(512),
    ElasticTransform(150),
    RandomCrop(512),
    HorizontalFlip(),
    Split([0, 3], [3, 6]),
    ], 
    PRNG, 
    border='constant', 
    fillval=0,
    anchor_index=3)
# image: np.ndarray of shape (h, w, 3), RGB format
# target: np.ndarray of shape (h, w, c)
transformed_image, transformed_target = transfrom(image, target)
```
![](https://i.loli.net/2018/01/06/5a5006c0d99b1.gif)  
Note that the example augmentations above are just for demonstration, there is no warranty that they are useful.

### More
* For randomness part, we pass RandomState as class/function argument, instead of using global seeds. It's thread-safe;
* This project has few photometric transformations, but provides a `Lambda` class, you can inserted 3rd party photometric transformation functions into our pipeline by using `Lambda`.
* 

### TODO
* More transformations
* Docstring or documentation

### Contact 
If you have problems related to this project, you can report isseus, or email me (qihang@outlook.com).
