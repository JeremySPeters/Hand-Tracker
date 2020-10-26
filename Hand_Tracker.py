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
    cv2.createTrackbar('Rmin', slidermenu, 0, 255, nothing)
    cv2.createTrackbar('Gmin', slidermenu, 0, 255, nothing)
    cv2.createTrackbar('Bmin', slidermenu, 0, 255, nothing)

    cv2.createTrackbar('Rmax', slidermenu, 255, 255, nothing)
    cv2.createTrackbar('Gmax', slidermenu, 255, 255, nothing)
    cv2.createTrackbar('Bmax', slidermenu, 255, 255, nothing)

    cv2.createTrackbar('Area', slidermenu, 240, 500, nothing)

    while True:
        """
        Flips image
        """
        ret, img = feed.read()
        img = cv2.flip(img, 1)

        """
        Creates blurred image to reduce noise
        """
        blur = cv2.GaussianBlur(img, (5,5), 0)


        """
        Creates trackbars for RGB thresholds
        """

        #Min Thresholds
        min_r = cv2.getTrackbarPos('Rmin', slidermenu)
        min_g = cv2.getTrackbarPos('Gmin', slidermenu)
        min_b = cv2.getTrackbarPos('Bmin', slidermenu)

        #Max Thresholds
        max_r = cv2.getTrackbarPos('Rmax', slidermenu)
        max_g = cv2.getTrackbarPos('Gmax', slidermenu)
        max_b = cv2.getTrackbarPos('Bmax', slidermenu)

        lower_rgb = np.array([min_b, min_g, min_r])
        upper_rgb = np.array([max_b, max_g, max_r])

        rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        rgbmask = cv2.inRange(rgb, lower_rgb, upper_rgb)

        filtered = cv2.bitwise_and(blur, blur, mask=rgbmask)

        tarea = cv2.getTrackbarPos('Area', slidermenu)

        contours, _ = cv2.findContours(rgbmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (tarea*100):
                cv2.drawContours(filtered, contour, -1, (0, 255, 0), 8)


        #cv2.drawContours(img, contours, -1, (0,255,0), 3)
        #cv2.drawContours(filtered, contours, -1, (0, 255, 0), 3)


        if cv2.waitKey(1) == 27:
            break  # esc to quit
        #cv2.imshow(wname, img)
        cv2.imshow(wname, filtered)

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