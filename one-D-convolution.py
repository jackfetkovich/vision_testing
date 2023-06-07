import cv2
import numpy as np
import matplotlib.pyplot as plt

def identify_maxima(peaks, FLAT_THRESH = 200, HEIGHT_THRESH = 10000, WIDTH_THRESH = 2):
  
  # Find the maxes
    # * Find the slope between every point and it's neighbor
    # * Consider it to be zero within some threshold
    # * Define some minimum "width" for it to be considered
    # * For all regions meeting constraints, identify the middle point as a max
  slopes = np.gradient(peaks)
  maxima = []

  i = 0
  while i < len(slopes):
    print(i)
    k = 0
    meeting_constraints = True
    num_points_meeting_constraints = 0
    
    while meeting_constraints and k <= i:
      if not(peaks[i + k] >= HEIGHT_THRESH and abs(slopes[i + k]) <= FLAT_THRESH):
        meeting_constraints = False
        break
      k+=1
      num_points_meeting_constraints +=1

    if k >= WIDTH_THRESH:
      maxima.append(int((i + k)- (k/2)))

    i+= k if k >=1 else 1

  return maxima

input = cv2.imread('./multiple_poles.png')
img = cv2.cvtColor(input, cv2.COLOR_BGR2HLS)

lowHLS = np.array([0, 52.4, 42.5]) # low color bound
highHLS = np.array([26.9, 255, 255]) # high color bound

mask = cv2.inRange(img, lowHLS, highHLS) # perform the mask
mask = cv2.medianBlur(mask, 7) # noise filtering

# 1D Convolution Time
cols = mask.shape[1]
rows = mask.shape[0]

peaks = np.zeros(cols)

for col in range(cols): # This is the actual convolution
    sum = 0
    for row in range(rows):
      sum += mask[row][col]
    peaks[col] = sum

maxima = identify_maxima(peaks)

for max in maxima:
  cv2.line(input, (max, 0), (max, mask.shape[0]), (255, 52, 175), thickness=2)

cv2.imshow("image", input)
cv2.waitKey(0)