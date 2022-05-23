
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


from moviepy.editor import VideoFileClip
from IPython.display import HTML
import queue
import pickle
import os
import sys
from scipy.ndimage.measurements import label
import glob
from skimage.feature import hog

orient = 9
pix_per_cell = 8
cell_per_block = 2


def convert_color(img, conv='RGB2YCrCb'):
    """
    Convert the image from one color space to the other
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'Gray':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)



def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    """
    Return the hog features of the given input image
    Call with two outputs if vis==True"""
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)
        return features

  # While it could be heavy to include three color channels of a full resolution image,
# you can perform spatial binning on an image and still retain enough information to help in finding vehicles.

# even going all the way down to 32 x 32 pixel resolution, 
# the car itself is still clearly identifiable by eye, and this means that the relevant features are still preserved at this resolution.

def bin_spatial(img, size=(16, 16)):         # to resize image combine each 4 pixel into 1 to make use of color info as a feature  
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features 
    # Compute the histogram of the RGB channels separately
    # Concatenate the histograms into a single feature vector

# NEED TO CHANGE bins_range if reading .png files with mpimg!                                                                  32 is the width for 1 region in histogram
def color_hist(img, nbins=32, bins_range=(0, 256)):              # to normalize histogram   1 Byte for pixel ranges [0-256]    256/32 = 8 --> 8 regions for histogram
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#Read cars and not-cars images

#Data folders
test_images_dir = './test_images/'

# images are divided up into vehicles and non-vehicles
test_images = []

images = glob.glob(test_images_dir + '*.jpg')

for image in images:
        test_images.append(mpimg.imread(image))

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, vis_bboxes = False):
    
    draw_img = np.copy(img)
    xstart = int(img.shape[1]/5) #from 1/5 of image 
    xstop = img.shape[1]         #till end of image 
    img_tosearch = img[ystart:ystop, xstart:xstop,:]  #crop image --> make it = size of rectangle
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
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
    rectangles = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() # ravel flaten the array 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).reshape(1, -1)
#             hog_features = np.hstack((hog_feat1))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
#             subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
 
             # Get color features
#             spatial_features = bin_spatial(subimg, size=spatial_size)
#             hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction
#             stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(hog_features)   
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or vis_bboxes == True:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))
                              
    return rectangles

def get_rectangles(image, scales = [1, 1.5, 2, 2.5, 3], 
                   ystarts = [400, 400, 450, 450, 460], 
                   ystops = [528, 550, 620, 650, 700]):
    out_rectangles = []
    X_scaler = pickle.load(open('X_scaler.pkl', 'rb'))
    svc = pickle.load(open('svc_pickle.pkl', 'rb'))
    spatial_size = 32
    hist_bins = 32
    for scale, ystart, ystop in zip(scales, ystarts, ystops):    #zip [(1, 400, 528), (1.5, 400, 550), ...]
        rectangles = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if len(rectangles) > 0:
            out_rectangles.append(rectangles)
    out_rectangles = [item for sublist in out_rectangles for item in sublist] 
    return out_rectangles

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap



def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    img_copy = np.copy(img)
    result_rectangles = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        area = (bbox[1][1] - bbox[0][1]) * (bbox[1][0] - bbox[0][0])
        if area > 40 * 40:
            result_rectangles.append(bbox)
            # Draw the box on the image
            cv2.rectangle(img_copy,(bbox[0][0]-int(0.02*bbox[0][0]),bbox[0][1]-int(0.02*bbox[0][1])), (bbox[1][0]+int(0.02*bbox[1][0]),bbox[1][1]+int(0.02*bbox[1][1])), (0,255,0), 6)
    # Return the image
    return result_rectangles, img_copy


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

class DetectionInfo():
    def __init__(self):
        self.max_size = 10
        self.old_bboxes = queue.Queue(self.max_size) 
        self.heatmap = np.zeros_like(test_images[0][:, :, 0])

    #added 
    def draw_heat_map(self, bboxes):
        heat_img = np.zeros_like(test_images[0][:, :, 0])
        heat_img = add_heat(heat_img, bboxes) #bbt3at numpy array of zeros wel return byb2a (Add += 1 for all pixels inside each bbox) y3ni ha5od el gowa el boxes bs ba2i el sora b zeros
        return heat_img

    #added
    def draw_all_boxes(self, image, bboxes):
        all_boxes = draw_boxes(image, bboxes, color='random', thick=3) 
        return all_boxes

    def get_heatmap(self):
        self.heatmap = np.zeros_like(test_images[0][:, :, 0])
        if self.old_bboxes.qsize() == self.max_size:
            for bboxes in list(self.old_bboxes.queue):
                self.heatmap = add_heat(self.heatmap, bboxes)
            self.heatmap = apply_threshold(self.heatmap, 20)
        return self.heatmap
    
    def get_labels(self):
        return label(self.get_heatmap())

    def add_bboxes(self, bboxes):
        if len(bboxes) < 1:
            return
        if self.old_bboxes.qsize() == self.max_size:
            self.old_bboxes.get()
        self.old_bboxes.put(bboxes)

detection_info = DetectionInfo()


def find_vehicles_debug(image):
    out_img = np.copy(image)
    bboxes = get_rectangles(image) 
    detection_info.add_bboxes(bboxes)
    
    all_boxes = detection_info.draw_all_boxes(out_img, bboxes) ###########

    heat_img = detection_info.draw_heat_map(bboxes) ############

    labels = detection_info.get_labels()
    if len(labels) == 0:
        result_image = image
    else:
        bboxes, result_image = draw_labeled_bboxes(image,labels)

    result = combine_images(result_image, all_boxes, heat_img)
    return result


def find_vehicles(image):
    bboxes = get_rectangles(image) 
    detection_info.add_bboxes(bboxes)
    labels = detection_info.get_labels()
    if len(labels) == 0:
        result_image = image
    else:
        bboxes, result_image = draw_labeled_bboxes(image,labels)

    return result_image

def combine_images( big_image, small1, small2):        
        
        background = np.zeros_like(big_image)
        large_img_size = (background.shape[1],int(0.75*background.shape[0])) 
        small_img_size=(int(background.shape[1]/2),int(0.25*background.shape[0]))
        small_img_x_offset=0 
        small_img_y_offset=0

        img2 = cv2.resize(np.dstack((small2, small2, small2))*255, small_img_size)
        img1 = cv2.resize(small1, small_img_size)
        img3 = cv2.resize(big_image, large_img_size)

        background[0: small_img_size[1], 0: small_img_size[0]] = img2
        
        start_offset_y = small_img_y_offset 
        endy = start_offset_y + small_img_size[1]
        if endy > background.shape[0]:
          endy = background.shape[0]
        start_offset_x = 2 * small_img_x_offset + small_img_size[0]
        endx = start_offset_x + small_img_size[0]
        if endx > background.shape[1]:
          endx = background.shape[1]
        background[start_offset_y:endy , start_offset_x: endx] = img1

        background[int(0.25*background.shape[0]): , : ] = img3
         
        return background


def create_video(in_path, out_path, mode):
  detection_info.old_heatmap = np.zeros_like(test_images[0][:, :, 0])
  project_video_path = in_path
  project_video_output = out_path

  project_video = VideoFileClip(project_video_path)
  if mode == '0' :
    white_clip = project_video.fl_image(find_vehicles) #NOTE: this function expects color images!!
    white_clip.write_videofile(project_video_output, audio=False)
  elif mode == '1':
    white_clip = project_video.fl_image(find_vehicles_debug) #NOTE: this function expects color images!!
    white_clip.write_videofile(project_video_output, audio=False)



def main():
    mode = sys.argv[3]
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    create_video(in_path,out_path,mode)

if __name__ == "__main__":
    main()

