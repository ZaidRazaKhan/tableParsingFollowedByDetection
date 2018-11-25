import numpy as np
import cv2 as cv
import utils
from table import Table
from PIL import Image
import xlsxwriter
import sys
from pdf2image import convert_from_path


if len(sys.argv) < 2:
    print("Usage: python driver.py <img_path>")
    sys.exit(1)


path = sys.argv[1]
if not path.endswith(".pdf") and not path.endswith(".jpg"):
    print("Image should be either in pdf or jpej")
    sys.exit(1)

if path.endswith(".pdf"):
    ext_img = convert_from_path(path)[0]
else: 
    ext_img = Image.open(path)

ext_img.save("data/target.jpg", "JPEG")
image = cv.imread("data/target.jpg")



# Checking whwther the given image is already converted into grayscale or not, if it is not converted into grayscale convert it
NUM_CHANNELS = 3  # FOR RGB
if len(image.shape) == NUM_CHANNELS:
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)



# Image Filtering using adaptive thresholding

# Constant variables for thresholding and contour
MAX_THRESHOLD_VALUE = 255
BLOCK_SIZE = 15
THRESHOLD_CONSTANT = 0

# Filtering the image using adaptive thresholding
filtered = cv.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)


"""
HORIZONTAL AND VERTICAL LINE ISOLATION


To isolate the vertical and horizontal lines, 

1. Set a scale.
2. Create a structuring element.
3. Isolate the lines by eroding and then dilating the image.

"""

# Setting up a scale for line isolation
SCALE = 15

# Isolate horizontal and vertical lines using morphological operations

# Making a copy for isolating horizontal and vertical lines
horizontal = filtered.copy()
vertical = filtered.copy()

horizontal_size = int(horizontal.shape[1] / SCALE)
# Creating structuring elements for horizontal structure
horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
utils.isolate_lines(horizontal, horizontal_structure)

vertical_size = int(vertical.shape[0] / SCALE)
# Creating structuring elements for vertical strucutre
vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))


# Isolating the lines by eroding and then dilating the image
utils.isolate_lines(vertical, vertical_structure)

# Finding mask for contour of table using just the horizontal and vertical
mask = horizontal + vertical

# oringinal image need not be preserved and heirarchy is not important for us.
(_, contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


# Finding table joints to detect rows and columns
intersections = cv.bitwise_and(horizontal, vertical)

# Get tables from the images
tables = [] # list of tables
for i in range(len(contours)):
    # Verify that region of interest is a table
    (rect, table_joints) = utils.table_detection(contours[i], intersections)
    if rect == None or table_joints == None:
        continue

    # Create a new instance of a table
    table = Table(rect[0], rect[1], rect[2], rect[3])
 
    # Get an n-dimensional array of the coordinates of the table joints
    joint_coords = []
    for i in range(len(table_joints)):
        joint_coords.append(table_joints[i][0][0])
    joint_coords = np.asarray(joint_coords)

    # Returns indices of coordinates in sorted order
    # Sorts based on parameters (aka keys) starting from the last parameter, then second-to-last, etc
    sorted_indices = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
    joint_coords = joint_coords[sorted_indices]

    # Store joint coordinates in the table instance
    table.set_joints(joint_coords)

    tables.append(table)

    #cv.rectangle(image, (table.x, table.y), (table.x + table.w, table.y + table.h), (0, 255, 0), 1, 8, 0)
    #cv.imshow("tables", image)
    #cv.waitKey(0)


# Performing OCR 
out = "bin/"
table_name = "table.jpg"
psm = 6
oem = 3
mult = 3

utils.mkdir(out)
utils.mkdir("bin/table/")

utils.mkdir("excel/")
workbook = xlsxwriter.Workbook('excel/tables.xlsx')

for table in tables:
    worksheet = workbook.add_worksheet()
    
    table_entries = table.get_table_entries()

    table_roi = image[table.y:table.y + table.h, table.x:table.x + table.w]
    table_roi = cv.resize(table_roi, (table.w * mult, table.h * mult))

    cv.imwrite(out + table_name, table_roi)

    num_img = 0
    for i in range(len(table_entries)):
        row = table_entries[i]
        for j in range(len(row)):
            entry = row[j]
            entry_roi = table_roi[entry[1] * mult: (entry[1] + entry[3]) * mult, entry[0] * mult:(entry[0] + entry[2]) * mult]

            fname = out + "table/cell" + str(num_img) + ".jpg"
            cv.imwrite(fname, entry_roi)

            fname = utils.run_textcleaner(fname, num_img)
            text = utils.run_tesseract(fname, num_img, psm, oem)

            num_img += 1
 
            worksheet.write(i, j, text)

workbook.close()
