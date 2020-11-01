"""
@Author: Jeremy Peters


"""
import numpy as np
import cv2
import math

def nothing(x):
    """
    Filler function to be used in trackbar creation
    :param x: N/A
    :return: NONE
    """
    pass

def finger_finder(contour,ghostframe):
    if len(contour) > 10:
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        if not isinstance(type(defects), type(None)):
            fingers = 0
            for i in range(defects.shape[0]):
                ds, de, df, dd = defects[i][0]
                start, end, far = tuple(contour[ds][0]), tuple(contour[de][0]), tuple(contour[df][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if angle <= math.pi / 2:
                    fingers += 1
                    cv2.circle(ghostframe, far, 8, (211, 84, 0), -1)
            return fingers
    else:
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
    """
    cv2.createTrackbar('Hmin', slidermenu, 0, 180, nothing)
    cv2.createTrackbar('Smin', slidermenu, 0, 255, nothing)
    cv2.createTrackbar('Vmin', slidermenu, 0, 255, nothing)

    cv2.createTrackbar('Hmax', slidermenu, 180, 180, nothing)
    cv2.createTrackbar('Smax', slidermenu, 255, 255, nothing)
    cv2.createTrackbar('Vmax', slidermenu, 255, 255, nothing)
    """

    cv2.createTrackbar('Hmin', slidermenu, 14, 180, nothing)
    cv2.createTrackbar('Smin', slidermenu, 77, 255, nothing)
    cv2.createTrackbar('Vmin', slidermenu, 104, 255, nothing)

    cv2.createTrackbar('Hmax', slidermenu, 176, 180, nothing)
    cv2.createTrackbar('Smax', slidermenu, 158, 255, nothing)
    cv2.createTrackbar('Vmax', slidermenu, 255, 255, nothing)

    cv2.createTrackbar('Blur', slidermenu, 5, 50, nothing)

    while True:
        """
        Flips image
        """
        ret, img = feed.read()
        img = cv2.flip(img, 1)

        """
        Creates blurred image to reduce noise
        Kernal size for cv2 blur must always be an odd number, hence the shenanigans with variable 'b'
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

        """
        Threshold arrays
        """
        lower_HSV = np.array([min_H, min_S, min_V])
        upper_HSV = np.array([max_H, max_S, max_V])

        """
        Making an HSV mask, and applying HSV thresholds
        """
        HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask_HSV = cv2.inRange(HSV, lower_HSV, upper_HSV)

        """
        Applies blur filter created earlier
        """
        cv2.bitwise_and(blur, blur, mask=mask_HSV)

        """
        Further refine image using erode and dilate
        """
        k = 2
        kernal = np.ones((k,k), np.uint8)
        cv2.erode(mask_HSV, kernal,cv2.CV_8UC1)
        cv2.dilate(mask_HSV, kernal,cv2.CV_8UC1)

        """
        Contours:
        Within the loop, it tries to find the largest contoured area 
        After finding the largest contoured area, it then assigns the actual area to area[0], and the contour array to area[1]
        """
        contours, hierarchy = cv2.findContours(mask_HSV, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            area = [-1, []]
            for contour in contours:
                con = cv2.contourArea(contour)
                if con > area[0]:
                    area[0] = con
                    area[1] = contour
            hull = cv2.convexHull(area[1])
            ghostframe = np.zeros(img.shape, np.uint8) #Will never be displayed
            cv2.drawContours(img, [area[1]], 0, (0, 255, 0), 2) #Hand outline
            cv2.drawContours(img, [hull], 0, (0, 0, 255), 3) #Hand wireframe

            """
            Fingers:
            - Gets number of fingers and then displays them using GPIO and LEDs
            - finger_finder is error prone due to an issue with openCV's "convexityDefects" function, best to ignore
            """
            fingers = 0
            try:
                fingers = finger_finder(area[1], img)
            except:
                pass
            if not isinstance(type(fingers), type(None)):
                    if fingers == 0:
                        pass
                    if fingers == 1:
                        pass
                    if fingers == 2:
                        pass
                    if fingers == 3:
                        pass
                    if fingers >= 4:
                        pass

        """
        Shows image to make adjustments easier to manage
        """
        cv2.imshow(wname, img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit


    feed.release()
    cv2.destroyAllWindows()

def main():
    feed = cv2.VideoCapture(0)
    feed.set(3, 640)
    feed.set(4, 480)
    feed.set(cv2.CAP_PROP_FPS, 30)
    handler(feed, 'frame')
    exit(0)

if __name__ == '__main__':
    main()