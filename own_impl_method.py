# import the necessary packages
from scipy.spatial import distance as dist 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of images")
args= vars(ap.parse_args())


# Initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# and to store the images themselves
index = {}
images = {}

# loop over the images paths
for imagePath in glob.glob(args["dataset"] + "\*.jpg"):
    # extract the image filename (assumed to be unique) and 
    # load the image, updating the images dictionnary
    filename = imagePath[imagePath.rfind("\\") + 1:]
    image = cv2.imread(imagePath)
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update the index

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index[filename] = hist


# METHOD #3: ROLL MY OWN METHOD

def chi2_distance(histA, histB, eps = 1e-10):
    # Compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) /  (a + b + eps) for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d


# initialize the resutls dictionary
results = {}

# loop over the index
for (k, hist) in index.items():
    # compute the distance between the two histograms
    # uing the custom chi squared method, the update
    # the results dictionary
    d = chi2_distance(index["1.jpg"], hist)
    results[k] = d

# sort the resutls
results = sorted([(v, k) for (k, v) in results.items()])

# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images["1.jpg"])
plt.axis("off")

# initialize the results figure
fig = plt.figure("Results: Custom Chi-Squared")
fig.suptitle("Custom Chi-Squared", fontsize = 20)

# loop over the results
for (i, (v, k)) in enumerate(results):
    # show the result
    ax = fig.add_subplot(1, len(images), i + 1)
    ax.set_title("%s: %.2f" % (k, v))
    plt.imshow(images[k])
    plt.axis("off")

# Show the custom methods
plt.show()