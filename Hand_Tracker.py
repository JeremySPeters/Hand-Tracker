"""
@Author: Jeremy Peters


"""
import numpy as np
import cv2

def nothing(x):
    """
    Filler function to be used in trackbar creation
    :param x: N/A
    :return: NONE
    """
    pass

def handler(feed, wname):
    """

    :param feed: Video Source
    :param wname: Main window name for displaying video feed
    :return: NONE
    """
    slidermenu = 'adjustments'

    cv2.namedWindow(wname)
    cv2.namedWindow(slidermenu)


    font = cv2.FONT_HERSHEY_PLAIN
    cv2.createTrackbar('Hmin', slidermenu, 0, 180, nothing)
    cv2.createTrackbar('Smin', slidermenu, 0, 255, nothing)
    cv2.createTrackbar('Vmin', slidermenu, 0, 255, nothing)

    cv2.createTrackbar('Hmax', slidermenu, 180, 180, nothing)
    cv2.createTrackbar('Smax', slidermenu, 255, 255, nothing)
    cv2.createTrackbar('Vmax', slidermenu, 255, 255, nothing)

    cv2.createTrackbar('Blur', slidermenu, 1, 50, nothing)
    cv2.createTrackbar('Area', slidermenu, 240, 500, nothing)

    while True:
        """
        Flips image
        """
        ret, img = feed.read()
        img = cv2.flip(img, 1)

        """
        Creates blurred image to reduce noise
        kernal size for cv2 blur must always be an odd number, hence the shenanigans with variable 'b'
        """
        b = (cv2.getTrackbarPos('Blur', slidermenu) * 2) + 1
        blur = cv2.GaussianBlur(img, (b,b), 0)

        """
        Creates trackbars for HSV thresholds
        """

        #Min Thresholds
        min_H = cv2.getTrackbarPos('Hmin', slidermenu)
        min_S = cv2.getTrackbarPos('Smin', slidermenu)
        min_V = cv2.getTrackbarPos('Vmin', slidermenu)

        #Max Thresholds
        max_H = cv2.getTrackbarPos('Hmax', slidermenu)
        max_S = cv2.getTrackbarPos('Smax', slidermenu)
        max_V = cv2.getTrackbarPos('Vmax', slidermenu)

        lower_HSV = np.array([min_H, min_S, min_V])
        upper_HSV = np.array([max_H, max_S, max_V])

        HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask_HSV = cv2.inRange(HSV, lower_HSV, upper_HSV)

        blur_HSV = cv2.bitwise_and(blur, blur, mask=mask_HSV)

        #area = cv2.getTrackbarPos('Area', slidermenu)

        #contours, _ = cv2.findContours(rgbmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        """for contour in contours:
            area = cv2.contourArea(contour)
            if area > (tarea*100):
                cv2.drawContours(filtered, contour, -1, (0, 255, 0), 8)"""


        #cv2.drawContours(img, contours, -1, (0,255,0), 3)
        #cv2.drawContours(filtered, contours, -1, (0, 255, 0), 3)


        if cv2.waitKey(1) == 27:
            break  # esc to quit
        #cv2.imshow(wname, img)
        cv2.imshow(wname, blur_HSV)

    feed.release()
    cv2.destroyAllWindows()

def main():
    feed = cv2.VideoCapture(0)
    feed.set(3, 640)
    feed.set(4, 480)
    handler(feed, 'frame')
    exit(0)

if __name__ == '__main__':
    main()