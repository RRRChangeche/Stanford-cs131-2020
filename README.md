# CS131: Computer Vision Foundations and Applications

This repository contains the released assignment for the fall 2020 of CS131, a course at Stanford taught by [Juan Carlos Niebles](http://www.niebles.net/) and [Ranjay Krishna](http://ranjaykrishna.com/index.html).

## Assignments

* Environment: python3.6+
* Requirements: run     "pip install -r requirements.txt"

## hw0: [[link]](https://github.com/RRRChangeche/Stanford_CS131_2020/tree/main/fall_2020/hw0_release)

* Linear Algebra and Numpy review
* Image manipulation
  * laod image
  * crop image
  * image dimmer
  * resize image
  * Rotating 2D coordinates
  * Rotate image
  
| load image | blur | rotate |
| ------------- | ------------- | ------------- |
| ![0_1](fall_2020/hw0_release/image1.jpg)  | ![0_2](fall_2020/hw0_release/16_16_baboon.png)  | ![0_3](fall_2020/hw0_release/rotated_output.png)  |

## hw1: [[link]](https://github.com/RRRChangeche/Stanford_CS131_2020/tree/main/fall_2020/hw1_release)

* Convolution
  * prove Commutative property
  * prove Shift invariance
  * prove Linearity
  * Implementation
    * conv_nested - using 4 for-loops
    * conv_fast - using zero-pad and numpy dot product
      1. zero padding
      2. flip kernel vertically and horizontally
      3. compute weighted sum

* Cross-correlation
  * Template matching
  * Zero-mean cross-correlation
  * Normalized cross-correlation

* Separable Filter
  * Theory
  * Complexity comparison

> What I've learned?
>
> * Optimize convolution implementaion by using numpy(conv_fast).
> * It's about up to 15x faster than naive implementation(conv_nested) in this case.
>
> * The `Zero-mean Cross-Correlation` is not robust to change in lighting condition.
>
> * Implement template matching by `Normalized Cross-Correlation` algorithm to search the target pattern in the source image.
>
> * 2D separable convolution is equivalent to two 1D convolutions.
>
> * Optimize 2D convolution by doing two 1D convolutoins.
>
> * It's about 2x faster by using separable filter in this case.

|**Convolution**|
|-|
|![1_4](fall_2020/hw1_release/1_4_output.png)|
|**Zero-mean cross-correlation**|
|![2_1](fall_2020/hw1_release/2_1_output.png) |
|**Normalized cross-correlation**|
|![1_4_3](fall_2020/hw1_release/2_2_output.png) |
|**Normal V.S. Separable convolution**|
![3_1](fall_2020/hw1_release/3_1_output.png)

## Recitation 4 - Advanced Numpy Tutorial: [[link]](https://github.com/RRRChangeche/Stanford_CS131_2020/tree/main/fall_2020/hw1_release)

* Creating numpy array
* Array Attributes
* Accessing elements
* Matrix operations
  * Element-wise operation - all arithmetic operations are applied to a matrix element-wise
    * +, -, *, /
    * sqrt/ power
  * Axis-based operations
    * mean
    * std
    * sum
  * Some other useful manipulations
    * concatenate
    * tranpose
    * linalg.inv - to invert a matrix 
* Numpy for linear algebra
  * Dot product - np.dot - return scalar
  * np.matmul/ @
* Numpy arrays, references, and *copies*
  * import copy/ copy.copy() and copy.deepcopy()
  * np.array.copy()
* Sorting
  * np.sort - return sorted array
  * np.argsort - return sorted indices
* Reshaping 
* Broadcsting
* Using boolean masks
  * boolean masks
  * np.where
* Linear Algebra
  * np.linalg.slove
  * np.linalg.lstsq
  * $\theta = (X^T X)^{-1} X^T y$
* Vectorize equations

> What I've learned?
>
> * All arithmetic operations are applied to a matrix element-wise.
>
> * Numpy operation is way faster than list.
>
> * Reference of numpy array, point to the same object. Use `copy` to fix this. 
>
> * Be aware of `copy` and `deepcopy`.
>
> * Broadcasting - Two dimensions are compatible when `they are equal`, or `one of them is 1`. [(details)](https://numpy.org/doc/stable/user/basics.broadcasting.html)
> ex: c = a*b, when a's shape is (1,3)/ b's shape is (3,1), then c's shape is (3,3) because of broadcasting  operation.
>

## hw2: [[link]](https://github.com/RRRChangeche/Stanford_CS131_2020/tree/main/fall_2020/hw2_release)

* Canny edge detector
  1. Smoothing
  2. Finding gradients
  3. Non-maximum suppression
  4. Double thresholding
  5. Edge tracking by hysterisis
  6. Optimize parameters
* Lane detection
  * Edge detection
  * Extracting Region of Interest (ROI)
  * Fitting lines using Hough Transform

> What I've learned?
>
> * **Implement Gaussian filter**
>
> * **Implement Gradient and calculate it's magnitude and direction of each pixel.**
>
> * **NMS流程**
>   * 先將方向以45度分為8等分 [0,45,90,135,180,225,270,315]
>   * 在梯度方向和鄰近像素比較梯度大小
>   * 如果像素在鄰近像素中是梯度最大, 則將像素留下
>
> * **Hough Transform steps:**
>   * Find edges using Canny edge detector
>   * Map edge points to Hough Space (r-theta coordinate)
>   * Store mapped points in accumulator
>   * Interpretate the accumulator to yield lines of infinite length
>   * Converse infinite lines to finite lines
>   * [Reference](https://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf)

|Smooth|
|-|
|![1_1](fall_2020/hw2_release/1_1_output.png) |
|**Grdient of x and y**|
|![1_2](fall_2020/hw2_release/1_2_output.png) |
|**NMS**|
|![1_3](fall_2020/hw2_release/1_3_output.png) |
|**Double Thresholding**|
|![1_4](fall_2020/hw2_release/1_4_output.png) |
|**Edge tracking by hysterisis**|
|![1_5](fall_2020/hw2_release/1_5_output.png) |
|**Edge detection**|
|![2_1](fall_2020/hw2_release/2_1_output.png) |
|**ROI**|
|![2_2](fall_2020/hw2_release/2_2_output.png) |
|**Hough transform**|
|![2_3](fall_2020/hw2_release/2_3_output.png) |

## hw3: [[link]](https://github.com/RRRChangeche/Stanford_CS131_2020/tree/main/fall_2020/hw3_release)

* Harris corner detector
* Describing and Matching Keypoints
* Transformation Estimation 計算轉置矩陣
* RANSAC
* Histogram of Oriented Gradients (HOG)
* Better Image Merging - Linear Blending
* Stitching Multiple Images

> What I've learned?
>
> * **Harris corner detector**
>   * A method to get keypoints of the image.
>   * Steps:
>   1. Compute Ix and Iy derivatives of an image.
>   2. Compute product of Ix^2/ Iy^2/ IxIy at each pixel
>   3. Apply Gaussian smooth to 2. to get Sxx/ Syy/ Sxy
>   4. Compute matrix M at each pixel and eigen value of M
>   5. Compute response value R at each pixel, where R = Det(M)-k(Trace(M)^2)
>   6. Use function `corner_peaks` from `skimage.feature` to get local maxima of response map by performing NMS and **localize keypoints**
>
> * **Describing and Matching Keypoints**
>   * Creating descriptors
>     * Give each keypoints a patch
>     * Normalize patch and then flattening into a 1D array
>   * Matching descriptors
>	    * Calculate Euclidean distance between all pairs of descriptors form image1 to image2 (use `scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')`)
>	    * Store 'match descriptors', if (first-closest pairs/ second-closest pairs) < threshold, then first-closest pairs is match
>
> * **Transformation Estimation**
>   * Use least square to calculate affine transformation matrix H letting p2*H = p1 (use `np.linalg.lstsq`)
>
> * **RANSAC**
>   * Steps:
>   1. Given parameters: 
>     - n_iters: 重複疊代次數
>     - threshold: 判斷inliers閥值
>   2. 隨機選取一組配對點 Select random set of matches
>   3. 計算仿射變換矩陣 使p2*H = p1 Compute affine transformation matrix (use `np.linalg.lstsq`)
>   4. 以歐式距離根據給定threshold判斷inliers數量 Compute inliers via Euclidean distance (use `np.linalg.norm(...ord=2)`)
>   5. 將有最多inliers的模型保存起來 Keep the largest set of inliers
>   6. 使用最多的那些inliers 重新計算一次Least-squares Re-compute least-squares estimate on all of the inliers
>
> * **Histogram of Oriented Gradients (HOG)**
>   * A method to describe keypoint
>   * Steps:
>   1. Compute the gradient image in x and y directions
>      * Use the sobel filter provided by skimage.filters
>   2. Compute gradient histograms
>      * Divide image into cells, and calculate histogram of gradients in each cell
>   3. Flatten block of histograms into feature vector
>   4. Normalize flattened block by L2 norm
>
> * **Better Image Merging - Linear Blending**
>   * Define left and right margins for blending to occur between
>   * Define a weight matrix for image 1 such that:
> 	  * From the left of the output space to the left margin the weight is 1
>     * From the left margin to the right margin, the weight linearly decrements from 1 to 0
>   * Define a weight matrix for image 2 such that:
> 	  * From the right of the output space to the right margin the weight is 1
> 	  * From the left margin to the right margin, the weight linearly increments from 0 to 1
>   * Apply the weight matrices to their corresponding images
>   * Combine the images
>
> * **Stitching Multiple Images**
>   * Combine the effects of multiple transformation matrices

|Harris corner response|
|-|
|![1_1](fall_2020/hw3_release/1_output.png) |
|**Keypoinnts**|
|![1_2](fall_2020/hw3_release/1_2_output.png) |
|**Describing and matching keypoints (simple descriptor)**|
|![2](fall_2020/hw3_release/2_output.png) |
|**Transformation matrix $H$ with `least square` method**|
|![3_1](fall_2020/hw3_release/3_1_output.png) |
|**Transformation matrix $H$ with `RANSAC` method (Robust matches)**|
|![3_2](fall_2020/hw3_release/3_2_output.png) |
|![3_3](fall_2020/hw3_release/3_3_output.png) |
|**HOG descriptor**|
|![4_2](fall_2020/hw3_release/4_2_output.png) |
|![4_3](fall_2020/hw3_release/4_3_output.png) |
|**Better Image Merging - Linear Blending**|
|![5](fall_2020/hw3_release/5_output.png) |
|**Stitching Multiple Images**|
|![6](fall_2020/hw3_release/6_output.png) |
