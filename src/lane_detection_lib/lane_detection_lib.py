import cv2
import numpy as np


def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.int0(np.array([tl, tr, br, bl]))

def roi(x, y):
    return np.array(
        [[[x/4, y/2.4], [x-x/4, y/2.4], [x, y], [0, y]]], np.int32)

def masking(image):
    # polygon = np.array([[[0,1080], [0,0], [1920,0], [1920,1080]]], np.int32)#mask for igvc pic
    # polygon = np.array([[500,1100], [850,250], [1100,250], [1600,1100]])#for saharsh img
    if len(image.shape) == 3:
        (y, x, ch) = image.shape
    else:
        (y, x) = image.shape

    polygon = roi(x, y)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.fillPoly(mask, polygon, 1)
    cv2.polylines(mask, [polygon], True, (0, 0, 0), thickness=10)

    masked = cv2.bitwise_and(image, image, mask=mask)

    return masked


def resize(image, scale_percent):
    # changing the size of the image to display it in the given frame
    # scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resizing the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def mask_lane(image):
    """
    Apply color selection to the HSL images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    # Convert the input image to HSL
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # White color mask50
#    lower_threshold = np.uint8([0, 50, 15])
    lower_threshold = np.uint8([0, 100, 0])
    upper_threshold = np.uint8([255, 255, 255])
#    upper_threshold = np.uint8([70, 255, 255])

    ## TEMPORARY FOR BLUE LINES
#    lower_threshold = np.uint8([0, 0, 80])
#    upper_threshold = np.uint8([70, 100, 255])
 
 
    ## TEMPORARY FOR RED LINES
#    lower_threshold = np.uint8([110, 50, 70])
#    upper_threshold = np.uint8([170, 100, 150])


    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # red color mask
    lower_threshold = np.uint8([0, 0, 170])
    upper_threshold = np.uint8([255, 255, 255])
    red_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, red_mask)
#    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def perspective_change(frame):
    if len(frame.shape) == 3:
        y, x, ch = frame.shape
    else:
        y, x = frame.shape
    #pts1 = np.float32([[x/2.5,0], [x-x/2.5,0], [x,y], [0, y]])
    #pts1 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])
    pts1 = np.float32([[x/4, y/3], [x-x/4, y/3], [0, y], [x, y]])
#    pts1 = roi(x, y)
    pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(frame, M, (x, y))

    return dst

def inverse_perspective_change(frame):
    if len(frame.shape) == 3:
        y, x, ch = frame.shape
    else:
        y, x = frame.shape
    #pts1 = np.float32([[x/2.5,0], [x-x/2.5,0], [x,y], [0, y]])
    #pts1 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])
    pts1 = np.float32([[x/4, y/3], [x-x/4, y/3], [0, y], [x, y]])
#    pts1 = roi(x, y)
    pts2 = np.float32([[0, 0], [x, 0], [0, y], [x, y]])

    M = cv2.getPerspectiveTransform(pts2, pts1)
add
    dst = cv2.warpPerspective(frame, M, (x, y))

    return dst

def get_lines(frame):
    #mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = masking(frame)
    mask = mask_lane(mask)
    frame = perspective_change(frame)
    mask = perspective_change(mask)
    empty = np.zeros_like(mask)

    Gaussian = cv2.GaussianBlur(mask, (9, 9), 0)

    gray_img = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(np.uint8(gray_img), 50, 100, apertureSize=3)

    lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180,
                            threshold=120, minLineLength=0, maxLineGap=150)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (y1 > 400 or y2 > 400):  # Filter out the lines in the top of the image
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 30)
                cv2.line(empty, (x1, y1), (x2, y2), (0, 0, 255), 30)

    return frame, mask, empty

def get_contours(frame, image):
    if len(frame.shape) == 3:
        height, width, ch = frame.shape
    else:
        height, width = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.erode(frame, kernel, iterations=3)
    frame = cv2.dilate(frame, kernel, iterations=7)
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    angs = 0
    lc = 0
    lane_found = False
    if len(contours) > 0:
        lane_found = True
        #cv2.drawContours(image, contours, -1, (0,255,255), 3)
        for contour in contours:
            blackbox = cv2.minAreaRect(contour)
            blackbox = cv2.boxPoints(blackbox)
            blackbox = np.int0(blackbox)
            blackbox = order_points_new(blackbox)

            x, y, w, h = cv2.boundingRect(contour)
            if w>70 and h>70:
# making sure the contour is big enough to ensure it's not random noise
#                print(width, height)
#                       1920  1080

                cv2.polylines(image, [blackbox], True, (0, 255, 0), 3)

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)

                [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
                lefty = int((-x*vy/vx) + y)
                righty = int(((width-x)*vy/vx)+y)
                cv2.line(image,(width-1,righty),(0,lefty),(0,255,0),2)

                y_axis = np.array([0, 1])    # unit vector in the same direction as the y axis
                your_line = np.array([vx, vy])  # unit vector in the same direction as your line
                dot_product = np.dot(y_axis, your_line)
                ang = np.arcsin(dot_product)
                ang *= 180/np.pi
                if ang < 0:
                    ang += 180
                angs += ang

                lc += 1
    else:
        print("can't find lanes")
    if angs == 0:
        angs = 90.0
    
    if (not lane_found) or lc == 0:
        lc = 1
    print(angs/lc)
    return image, angs/lc, lane_found

def parse_image(frame):
    lines = get_lines(frame)
    blurred = cv2.GaussianBlur(lines[1], (17, 17), 0)
    return get_contours(blurred, lines[0])

if __name__ == "__main__":
    vid = cv2.VideoCapture(6)
    ret, frame = vid.read()
    while ret:
        ret, frame = vid.read()
        if ret:
            fr = get_lines(frame)
            ff = cv2.GaussianBlur(fr[1],(17,17),0)
            fr2, angs, _ = get_contours(fr[2], fr[0])
#            print(angs)
#            cv2.imshow('mask', resize(fr2, 50))
            cv2.imshow('mask', resize(fr[1], 50))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  

    vid.release()
    cv2.destroyAllWindows()
