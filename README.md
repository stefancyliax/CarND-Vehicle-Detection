# Udacity Project 5: Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project is to detect vehicles in a video recorded on a highway. Instead of using a modern deep learning approach like [Yolo](https://arxiv.org/abs/1512.00567), we were were given tools from classic computer vision like [HOG](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) and machine leaning like [Support Vector Machines](http://scikit-learn.org/stable/modules/svm.html).    

In the end the model was able to reliably and precisely detect the vehicle in front of the car. This document describes my approach and implemented pipeline.

## Approach

The project consists of two parts. The first is training a classifier on a provided dataset to classify between images of vehicles and "not-vehicles". The second is using the classifier to find cars in the provided video and annotate them in the video.

Project video: (Youtube link)

[![Project track](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/project_vid.gif)](https://youtu.be/-Tn7427VUVM)
Video with debugging output: (Youtube link)

[![Project track](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/debug_vid.gif)](https://youtu.be/dlsnkgHiDrg)

## Detailed explanation of the pipeline
### Training of Machine Learning classifier
For the training of the classifier a dataset of images were provided of both cars and "not-cars".

| Class      | Count  | Shape   |
|------------|--------|---------|
| Car        | 8792   | 64x64x3 |
| Not-car    | 8968   | 64x64x3 |

![dataset](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/dataset.png)


To train the classifier the following steps were taken:
1. Feature extraction
2. Normalize and splitting into Train and Test set
3. Principal Component Analysis (PCA)
4. Fitting of Support Vector Machine classifier

##### 1. Feature extraction
Every image consists of 64x6x3 = 12288 pixels that could used as features to train a classifier. This would be quite expensive though. Instead we are extracting histogram, color and gradient features from the image. This enables us to work with a much smaller feature vector (in this case 2544) while not losing to much information. This enables a effective use of an SVM classifier later.

The gradient features are extracted using [sklean.hog](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html). Histogram of Oriented Gradients (HOG) subsamples the image to a grid of e.g. 8x8 or 4x4 grids, that each only contain the most prominent gradient in that part of the image. It therefore extracts gradients in the image while dropping all other information.

![hog](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/Hog.png)

```python
# Helper function to extract HOG features and optional visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
    return features
```

For histogram features are extracted using [numpy.histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html). The histogram features extract the tonal distribution of the image.


```python
# Helper function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features
```

The color features are extracted by downscaling the image and taking the pixels as feature vector. In the downscaling process we reduce the number of pixel significantly but maintain the rough color information of the image.

TODO: insert image
```python
# Helper function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Resize image and flatten it
    features = cv2.resize(img, size).ravel()
    return features
```

##### 2. Normalize and splitting into Train and Test set
Next the features are normalized using [sklearn.preprocessing.StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and then split into a train and test set using [sklearn.model_selection.train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
The normalizing is needed since the features vectors are numerically way bigger than other feature vectors, as seen in the image below. After the normalization, the feature vector is distributed equally.

![scaler](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/Scaler.png)

```python
# Helper function to create dataset and normalize the features
def create_and_scale_trainset(car_features, notcar_features):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    return X_scaler, X_train, X_test, y_train, y_test
```

##### 3. Principal Component Analysis (PCA)
Next PCA was applied using [_sklearn.decomposition.PCA_](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Principal Component Analysis is a technique to reduce the number of dimensions of a feature vector while maintaining the information. PCA is able to [extract orthogonal components that explain a maximum amount of the variance](http://scikit-learn.org/stable/modules/decomposition.html#pca). Simply put, it allowed me to shrink the feature vector from 2544 to 150 with only a minor loss in accuracy. Because the prediction time is highly dependent on the number of features, I was now able to use a more expensive _rbf_ kernel  for the Support Vector Machine kernel.

For the project, PCA allowed me to reduce the prediction time by a factor of ~10 while improving the accuracy by a factor of ~2.5.


|                     | Classifier  | PCA  | Accuracy | Prediction time for 100 samples |
|---------------------|-------------|------|----------|---------------------------------|
| First parameter set | linear, C=1 | None | 0.9913   | 0.2896s                         |
| Using PCA           | rbf, C=10  | n=150 | **0.9969**   | **0.0249s**                 |

```python
def fit_PCA(X_train, X_test, y_train, y_test, n_components=150):
    # Extracting the top (n_components) eigenfeatures
    pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True).fit(X_train)

    #Projecting the input data on the eigenfaces orthonormal basis
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return pca, X_train_pca, X_test_pca
```

#### 4. Fitting of Support Vector Machine classifier
Lastly I fit the data to a [Support Vector Machine classifier](http://scikit-learn.org/stable/modules/svm.html) with a radial basis function kernel (_rbf_).
Using the PCA approach I acceived a accuracy of **0.9969** while keeping the computation very efficient.

```python
def fit_SVM(X_train, X_test, y_train, y_test, kernel='linear', C=1.0):
    # Instanziate Support Vector Machine classifier
    svc = SVC(kernel=kernel, C=C, gamma=0.005)

    svc.fit(X_train, y_train)

    # Log accuracy on test set
    accuracy = round(svc.score(X_test, y_test), 4)

    return svc, accuracy
```


### Parameter selection
Since this project involves a lot of parameters, I decided to implement a automatic parameter testing. I created a spreadsheet with each row containing a parameter set I want to test. I exported the spreadsheet as ```.csv``` so I could use [pandas](http://pandas.pydata.org/) to read in line by line and do feature extraction and fitting using the read in parameters. In the end I inserted differnt performance indicators like accuracy and predict time back into the spreadsheer using pandas. This allowed for an large number of parameters to be tested over night. The spreadsheet containing all 50 parameter sets I tested [can be found in the repo](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/extraction_parameters.csv).

In the end I settled for the follwing parameter set.
| Color Space | HOG           | Color   | Histogram | Classifier | PCA   | Accuracy | Prediction time for 100 samples |
|-------------|---------------|---------|-----------|------------|-------|----------|---------------------------------|
| YCrCb       | 16,16,2,'ALL' | (16,16) | 16        | rbf, C=10  | n=150 | 0.9969   | 0.0249s                         |


## Vehicle Detection Pipeline
The detection pipeline uses the classifier from part 1 to detect vehicle in a image and later video.

the pipeline consists of the following steps.
1. Calculate HOG feature over entire image and subsample with three different window sizes. Extract color and histogram features as well.
2. Normalize and apply PCA to combined feature vector.
3. Use classifier from part 1 to predict if there is a car in the subsample.
4. False positive rejection
5. Annotate found vehicle in input image by drawing a bounding box.

##### 1. Calculate HOG feature over entire image and subsample
For the first step I adopted a code chunk from the lesson material. It applies a smart trick to the extraction of HOG features. Instead of adjusting the size of the sliding window and then rescaling the subsample back to 64x64 pixel to calculate HOG features, the function is resizing the entire image and calculating HOG features over the entire image. So instead of a 96x96 sliding window that we would have to rescale to 64x64 to calculate HOG features, we scale the entire imge down by a factor of 1.5 and subsample with a window size of 64x64.

Color and historam features are extracted from the same 64x64 window in this step.

The function ```find_cars() ``` does the following:
1. Scale image to desired size
2. Calculate HOG features over whole image
3. Subsample in windows of 64x64 pixel
4. Calculate color and histogram features on same subsample
5. Combine feature vectors
6. Normalize and apply PCA
7. Predict using previously trained classifier
8. Return coordinates of subsamples with predicted cars

```python
def find_cars(img, ystart, ystop, scale, svc, pca, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    bbox_list = []
    # scale image
    img = img.astype(np.float32)

    # drop upper part of image
    img_tosearch = img[ystart:ystop, :, :]
    # convert color
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)

    # scale single picture hog to new size
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)


            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
 ```


##### 2. Normalize and apply PCA to combined feature vector
After combining the feature vectors, they are normalized using the scaler from part 1 and then I applied the PCA from part 1 to reduce the number of features to 150.

 ```python
           test = np.hstack((spatial_features, hist_features, hog_features))
           # Scale features and make a prediction
           test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
           test_features_pca = pca.transform(test_features)
```


##### 3. Use classifier from part 1 to predict if there is a car in the subsample
Finally the
 ```python

            test_prediction = svc.predict(test_features_pca)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox_list.append([(xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)])

    return bbox_list
```
In the pipeline  ```find_cars() ``` was applied with the scales values 1, 1.5 and 2 to use search windows of 64x64, 96x96 and 128x128. The positive prediction were combined.

##### 4. False positive rejection
Next I implemented a class that kept a history of the last 30 frames to only draw bounding boxes of the detection was viable over a few frames. This was needed to rejected false positives in the video stream.
I implemented it using the concept of heatmap and thresholding. To smooth things out, the sum over the last 30 heatmaps was calculated and thresholded to 30.
```python
class Heatmap():
    def __init__(self, history_depth=10, threshold=2):

        self.threshold = threshold
        self.history_depth = history_depth
        self.history = np.array([[],[]])
        self.current_heatmap = np.array([[],[]])
        self.smoothed_heatmap = np.array([[],[]])


    def add_heat(self, bbox_list):
        self.current_heatmap = np.zeros((720, 1280)).astype(np.uint8)
        for box in bbox_list:
            self.current_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        self.__update_history()


    def __update_history(self):
        # update history
        if self.history.size == 0:
            self.history = self.current_heatmap.reshape(1,720,1280)
        else:
            self.history = np.vstack((self.history, self.current_heatmap.reshape(1,720,1280)))

        # limit depth of history to 10 newest
        if self.history.shape[0] > self.history_depth:
            self.history = self.history[1:,:,:]

        # calculate sum over history
        self.smoothed_heatmap = np.sum(self.history,axis=0)

        # apply threshold to reject false positives
        self.smoothed_heatmap[self.smoothed_heatmap <= self.threshold] = 0
        self.smoothed_heatmap = np.clip(self.smoothed_heatmap, 0, 255)
```

![heatmap](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/heatmap.png)


##### 5. Annotate found vehicle in input image by drawing a bounding box

To figure out how many cars the heatmap shows, the function [scipy.ndimage.measurements.label](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) was used. The output of ```label()``` is then passed to a function that draws bounding boxes on the input image.

```python
def draw_labeled_bboxes(img, labels, color=(0,1,0)):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixel with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixel
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        ymin = np.min(nonzeroy)
        ymax = np.max(nonzeroy)
        xmin = np.min(nonzerox)
        xmax = np.max(nonzerox)
        bbox = ((xmin, ymin), (xmax, ymax))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'car'
        cv2.putText(img,text, (xmin,ymin-12), font, 1, (0,1,0),2)
    # Return the image
    return img
```

![debug](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/debug.jpg)

![result](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/result.png)






# Discussion
project with many parameters and a long computation time that prolonged the interation cycles to



Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You can submit your writeup in markdown or use another method and submit a pdf instead.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
