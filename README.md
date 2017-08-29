# Udacity Project 5: Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project is to detect vehicles in a video recorded on a highway. Instead of using a modern deep learning approach like [Yolo](https://arxiv.org/abs/1512.00567), we were were given tools from classic computer vision and machine leaning like [HOG](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)  and [Support Vector Machines](http://scikit-learn.org/stable/modules/svm.html).    

In the end the pipeline was able to reliably and precisely detect the vehicles in front of the car. This document describes my approach and implemented pipeline. The code can be found in the corresponding [Jupyter Notebook](https://github.com/stefancyliax/CarND-Vehicle-Detection/blob/master/P5_Vehicle_Detection.ipynb).

Project video: (Youtube link)

[![Project track](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/project_vid.gif)](https://youtu.be/-Tn7427VUVM)

Video with debugging output: (Youtube link)

[![Project track](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/debug_vid.gif)](https://youtu.be/dlsnkgHiDrg)



## Detailed explanation of the pipeline

The project consists of two parts. The first is training a classifier on a provided dataset to classify between images of vehicles and "not-vehicles".
The second is applying the classifier to find cars in the provided video and annotate them.


### Training of the classifier
For the training of the classifier a dataset of images is provided of both cars and "not-cars".

| Class      | Count  | Shape   |
|------------|--------|---------|
| Car        | 8792   | 64x64x3 |
| Not-car    | 8968   | 64x64x3 |

Examples:

![dataset](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/dataset.png)

To train the classifier the following steps were taken:
1. Feature extraction
2. Normalize and splitting into Train and Test set
3. Principal Component Analysis (PCA)
4. Fitting of Support Vector Machine classifier

#### 1. Feature extraction
Each image is made of 12288 pixel (64x64x3), that could be used directly as features to train a classifier. Doing so would be quite expensive though. Instead we are extracting histogram, color and gradient features from the images. This enables us to work with a much smaller feature vector (in this case 2544) while not losing too much information.

The gradient features are extracted using a technique called [Histogram of Oriented Gradients (HOG)](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html). Basically HOG subsamples the image to a grid of e.g. 8x8 or 4x4 grids, with each only containing the most prominent gradient in that part of the image. It therefore extracts gradients in the image while dropping other information like color.

![hog](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/Hog.png)

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
        features = hog(img, orientations=orient,
                       pixel_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
    return features
```

Histogram features are extracted using [```numpy.histogram```](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html). With histogram features extract the tonal distribution of the image while not being sensitive to aspects and orientations.

```python

def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features
```

Color features are extracted by downscaling the image and taking the pixel as feature vector. In the downscaling process we reduce the number of pixel significantly but maintain the rough color information of the image.

```python
def bin_spatial(img, size=(32, 32)):
    # Resize image and flatten it
    features = cv2.resize(img, size).ravel()
    return features
```

#### 2. Normalize and splitting into Train and Test set
Next the features are normalized using [```sklearn.preprocessing.StandardScaler```](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
The normalizing is needed because the features vectors are numerically very different from each other, as seen in the image below. After the normalization, the feature vector is distributed equally.

![scaler](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/Scaler.png)

After normalization the dataset is split into train and test set using [```sklearn.model_selection.train_test_split```](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

```python
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

#### 3. Principal Component Analysis (PCA)
Next PCA was applied using [```sklearn.decomposition.PCA```](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). _Principal Component Analysis_ is a technique to reduce the number of dimensions of a feature vector while maintaining the information. PCA is able to [extract orthogonal components that explain a maximum amount of the variance](http://scikit-learn.org/stable/modules/decomposition.html#pca). Simply put, it allowed to shrink the feature vector from 2544 to 150 with only a minor loss in accuracy. Because the prediction time is highly dependent on the number of features, I was now able to use a more expensive _rbf_ kernel for the Support Vector Machine kernel.

For the project, PCA allowed me to reduce the prediction time by a factor of ~10 while improving the accuracy by a factor of ~2.5.

|                     | Classifier  | PCA  | Accuracy | Prediction time for 100 samples |
|---------------------|-------------|------|----------|---------------------------------|
| First parameter set | linear, C=1 | None | 0.9913   | 0.2896s                         |
| Using PCA           | rbf, C=10  | n=150 | **0.9975**   | **0.0249s**                 |

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
Using the PCA approach I achieved a accuracy of **0.9975** while keeping the computation very efficient.

```python
def fit_SVM(X_train, X_test, y_train, y_test, kernel='linear', C=1.0):
    # Instantiate Support Vector Machine classifier
    svc = SVC(kernel=kernel, C=C, gamma=0.005)
    svc.fit(X_train, y_train)

    # Log accuracy on test set
    accuracy = round(svc.score(X_test, y_test), 4)

    return svc, accuracy
```


### Parameter selection
Since this project involves a lot of parameters, I decided to implement an automatic parameter testing.

I created a spreadsheet with each row containing a parameter set I wanted to test. I saved the spreadsheet as ```.csv``` and used [pandas](http://pandas.pydata.org/) and a few lines of code to iterate though it row by row. For each row the script feed all parameters to feature extraction and fitting. I inserted different performance indicators like accuracy and predict time back into the spreadsheet again using pandas. This allowed for an large number of parameters to be tested over night. The spreadsheet containing all 50 parameter sets I tested [can be found in the repo](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/extraction_parameters.csv).

```python
# Note: shortened
parameter_file = "extraction_parameters.csv"
para_pd = pd.read_table(parameter_file, sep=';')

for index, para_row in para_pd.iterrows():

    # Skip rows that are already calculated
    if para_row['accuracy'] != 0:
        continue

    kernel = para_row['kernel']
    color_space = para_row['color_space']
    # shortened!

    car_features = extract_features_from_dataset(cars)  # shortened
    notcar_features = extract_features_from_dataset(notcars)  # shortened

    para_pd.loc[index, 'time_to_extract_features'] = time_extract
    para_pd.loc[index, 'time_to_extract_features_100'] = round(time_extract*100/(len(car_features)+len(notcar_features)),6)

    X_scaler, X_train, X_test, y_train, y_test = create_and_scale_trainset(car_features, notcar_features)

    if pca:
        pca, X_train_pca, X_test_pca = fit_PCA(X_train, X_test, y_train, y_test, n_components=pca_num)
        svc, accuracy, time_fit, time_prediction = fit_SVM(X_train_pca, X_test_pca, y_train, y_test, kernel=kernel, C=C)
        para_pd.loc[index, 'features_vector_length'] = int(len(X_train_pca[0]))
    else:
        svc, accuracy, time_fit, time_prediction = fit_SVM(X_train, X_test, y_train, y_test, kernel=kernel, C=C)
        para_pd.loc[index, 'features_vector_length'] = int(len(X_train[0]))

    para_pd.loc[index, 'training_time'] = time_fit
    para_pd.loc[index, 'accuracy'] = accuracy   
    para_pd.loc[index, 'prediction_time_100'] = time_prediction

    # save pandas file back to hard drive
    para_pd.to_csv(parameter_file, sep=';', index=False)
```


In the end I settled for the following parameter set.

| Color Space | HOG           | Color   | Histogram | Classifier | PCA   | Accuracy | Prediction time for 100 samples |
|-------------|---------------|---------|-----------|------------|-------|----------|---------------------------------|
| YCrCb       | 16,16,2,'ALL' | (16,16) | 16        | rbf, C=10  | n=150 | **0.9975**   | **0.0249s**                         |

I was suprised to see the best performance with a HOG parameter ```pix_per_cell = 16``` because this results in a HOG grid of just 4x4. See example below. A more complex HOG feature extraction was both slower and worse performing rather strangely.

![HOG16](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/Hog16.png)



## Vehicle Detection Pipeline
The detection pipeline scans over an image and uses the trained classifier, normalizer and PCA from part 1 to detect vehicle in an image and later video.

The pipeline consists of the following steps.
1. Calculate HOG feature over entire image and subsample with three different window sizes. Extract color and histogram features as well.
2. Normalize and apply PCA to combined feature vector.
3. Use classifier from part 1 to predict if there is a car in the subsample.
4. False positive rejection
5. Annotate found vehicle in input image by drawing a bounding box.

#### 1. Calculate HOG feature over entire image and subsample
For the first step I adopted a code chunk from the lesson material. It applies a smart trick to the extraction of HOG features. Instead of adjusting the size of the sliding window and then rescaling the subsample back to 64x64 pixel to calculate HOG features, the function is resizing the entire image and calculating HOG features over the entire image. So instead of a 96x96 sliding window that we would have to rescale to 64x64 to calculate HOG features, we scale the entire image down by a factor of 1.5 and subsample with a window size of 64x64.

Color and histogram features are extracted from the same 64x64 window in this step.

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


#### 2. Normalize and apply PCA to combined feature vector
Every subsample feature vector is then normalized using the same scaler as for training in part 1. After that the PCA from part 1 is applied to, again, reduce the number of features to 150.

```python
           test = np.hstack((spatial_features, hist_features, hog_features))
           # Scale features and make a prediction
           test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
           test_features_pca = pca.transform(test_features)
```


#### 3. Use classifier from part 1 to predict if there is a car in the subsample
Finally the classifier I trained in part 1 is used to predict whether there is a car in a subsample or not. For every positive prediction the coordinates of the subsample are saved and appended to ```bbox_list```.
```python

            test_prediction = svc.predict(test_features_pca)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox_list.append([(xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)])

    return bbox_list
```
The pipeline applies ```find_cars() ``` applies three times with scales values of 1, 1.5 and 2 to search the image in windows of 64x64, 96x96 and 128x128. The positive prediction are combined.

![find1](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/find1.png)

![find2](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/find2.png)

![find3](https://github.com/stefancyliax/CarND-Vehicle-Detection/raw/master/output_images/find3.png)


#### 4. False positive rejection
Next I implemented a class that kept a history of the last 30 frames to only draw bounding boxes when the detection was viable over a few frames. This was done to reject false positives in the video stream.

I implemented it using the concept of heatmap and thresholding. To smooth things out, the sum over the last 30 heatmaps is calculated and thresholded to 30.

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


#### 5. Annotate found vehicle in input image by drawing a bounding box

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
This was a interessting project were I could resort to lessons from the Udacity course [Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120) I did a while back. In the end the pipeline was able to reliably detect the vehicles in the project video and annotate them. The false positive rejection proved to be effective as well. One downside of the approach using computer vision and machine learning is that it rather computationally expensive. The processing of the whole video took about 15 minutes which translates to 1.3 fps. Obviously way too slow for a real world application.

#### Use class for every vehicle to do proper tracking on screen
I am pretty happy with the machine learning part but the vehicle detection pipeline leaves room for improvement. One possible improvement is the use of a class to properly track every detected vehicle.

#### Use of NN or CNN
It would be interesting to benchmark a NN as classifier against the Support Vector Machine, especially regarding computation speed. Also a benchmark against a modern approach like Yolo or other object detection CNN would be very interesting.
