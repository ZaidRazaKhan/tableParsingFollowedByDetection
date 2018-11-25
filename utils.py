import cv2 as cv
import pytesseract as tess
from PIL import Image
import subprocess as s
import os
from subprocess import call
from luminoth import Detector, read_image, vis_objects\


# For applying Morphological operations
def isolate_lines(src, structuring_element):
	cv.erode(src, structuring_element, src, (-1, -1)) # makes white spots smaller
	cv.dilate(src, structuring_element, src, (-1, -1)) # makes white spots bigger

def take_detected_table(image_name_with_location_in_directory):
    #Read the image from the location provided by the user
    image = read_image(image_name_with_location_in_directory)

    # If no checkpoint specified, will assume `accurate` by default. In this case,
    # we want to use our specific checkpoint. The Detector can also take a config
    # object.
    detector = Detector(checkpoint='c3256rfvt675')
    # Checkpoint that we have created when we were doing the training part of Faster R-CNN model
    
    # For storing multiple table present inside a document image
    table = []
    """
    # Returns a dictionary with the detections. Basically a Json Object containging the details in order {
        "file": "image.jpg",
        "objects": [
        {"bbox": [294, 231, 468, 536], "label": "table", "prob": 0.9997},
        {"bbox": [534, 298, 633, 473], "label": "header", "prob": 0.4089},
        {"bbox": [224,213,456,878], "label": "image", "prob":0.6996}
     ]
    }
    """
    predicted_object = detector.predict(image)
    
    for object in predicted_object['objects']:
        if(object['label'] == table):
            table.append(object['bbox'])

    return table


def table_details(table , intersections):
    """
    Possible Table region comming after detection from the luminoth Faster R-CNN model trained for our specific dataset
    """
    possible_table_region = intersections[table[0]:table[2], table[1]:table[3]]
    
    (_,possible_table_joints, _) = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    
    
    """
    Determines the number of table joints in the image If less than 5 table joints, then the image is likely not a table
    """
    if len(possible_table_joints) < 5:
        return None, None
    
    rect = [table[0],table[1],table[2]-table[0], table[3]-table[1]]
    return rect, possible_table_joints



# Table detection

MIN_TABLE_AREA = 50 # min table area to be considered a table
EPSILON = 3 # epsilon value for contour approximation
def table_detection(contour, intersections):
    
    area = cv.contourArea(contour)

    if (area < MIN_TABLE_AREA):
        return (None, None)

    curve = cv.approxPolyDP(contour, EPSILON, True)
    rect = cv.boundingRect(curve) # format of each rect: x, y, w, h
    
    possible_table_region = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    (_, possible_table_joints, _) = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    """ 
    Determines the number of table joints in the image If less than 5 table joints, then the image is likely not a table
    """
    if len(possible_table_joints) < 5:
        return (None, None)

    return rect, possible_table_joints



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def showImg(name, matrix, durationMillis = 0):
    cv.imshow(name, matrix)
    cv.waitKey(durationMillis)

def run_textcleaner(filename, img_id):
    mkdir("bin/cleaned/")
    # Run textcleaner
    cleaned_file = "bin/cleaned/cleaned" + str(img_id) + ".jpg"
    s.call(["./textcleaner", "-g", "-e", "none", "-f", str(10), "-o", str(5), filename, cleaned_file])
    return cleaned_file

def run_tesseract(filename, img_id, psm, oem):
    mkdir("bin/extracted/")
    image = Image.open(filename)
    language = 'eng'
    configuration = "-psm " + str(psm) + " -oem" + str(oem)
    text = tess.image_to_string(image, lang=language, config=configuration)
    if len(text.strip()) == 0:
        configuration += " -c tessedit_char_whitelist=0123456789"
        text = tess.image_to_string(image, lang=language, config=configuration)

    return text
