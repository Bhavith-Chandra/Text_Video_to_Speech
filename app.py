import logging
logging.captureWarnings(True)
import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug import secure_filename
from PIL import Image
import pytesseract
from gtts import gTTS
import sys
import math
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt


# Initialize the Flask application
app = Flask(__name__, static_url_path = "", static_folder = "static")

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['mp4'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# Warp img2 to img1 using the homography matrix H
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    return output_img

# Binarize the image
def binarize(img):
	img = cv2.medianBlur(img,5)
	#ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	return img

# Stitch two images together
def stitch(img1, img2, min_match_count):

    # Initialize the SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    #sift = cv2.SIFT()

    # Extract the keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize parameters for Flann based matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # Initialize the Flann based matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Compute the matches
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Store all the good matches as per Lowe's ratio test
    good_matches = []
    for m1,m2 in matches:
        if m1.distance < 0.7*m2.distance:
            good_matches.append(m1)

    if len(good_matches) > min_match_count:
        src_pts = np.float32([ keypoints1[good_match.queryIdx].pt for good_match in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[good_match.trainIdx].pt for good_match in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        result = warpImages(img2, img1, M)
        return result
        #cv2.imshow('Stitched output', result)
        #cv2.waitKey()
        #cv2.imwrite('/home/abhay/Codes/OCR/mosaic/images/1.jpg',result)

    else:
        print "We don't have enough number of matches between the two images."
        print "Found only %d matches. We need at least %d matches." % (len(good_matches), min_match_count)

# Captures different frames from a given video
def frameCapture(filename):

    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()

    count = 0
    total = 1
    print '\n\n'

    while success:
      success,image = vidcap.read()
      if count%80==0 :
	    cv2.imwrite("images/%d.jpg" % total, image)     # save frame as JPEG file
	    print 'Capturing frame ' + str(total)
	    total+=1
      if cv2.waitKey(10) == 27:                     # exit if Escape is hit
        break
      count += 1

    print '\n\n'
    return total


# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        print file.filename
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        
        #print '\n\nI am Here\n\n'

        total = frameCapture('uploads/'+file.filename)

        min_match_count = 7
        img1 = cv2.imread('images/1.jpg',0)
        img1 = binarize(img1)
        img1 = imutils.resize(img1, width=1000)


        for i in range(2,(total)):
            print 'Stitching image ' + str(i)
            img2 = cv2.imread('images/'+str(i)+'.jpg',0)
            img2 = binarize(img2)
            img2 = imutils.resize(img2, width=1000)
            img1 = stitch(img1, img2, min_match_count)

        cv2.imwrite('images/0.jpg',img1)
        cv2.imwrite('static/0.jpg',img1)

        print '\n\nConverting Image to Text'

        string = pytesseract.image_to_string(Image.open('images/0.jpg'))
        print '\n\nOCR OUTPUT\n\n' + string + '\n\n'

        f = open("static/test.txt","w")
        f.write(string)
        f.close()

        string = '"{}"'.format(string)
        print 'Converting Text to Speech\n\n'
        tts = gTTS(text=string, lang='en')
        tts.save("static/tts.mp3");

        return render_template('index1.html')

        #print 'Playing audio\n\n'
        #os.system("mpg321 abhay.mp3 -quiet")
        #return string
        #return redirect(url_for('uploaded_file',filename=filename))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(
        host="127.0.0.3",
        port=int("3000"),
        debug=True
    )