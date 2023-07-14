import cv2 as cv

def get_ball_position(img, opt, original_img_=None):
    ret, thresh = cv.threshold(img, opt.brightness_thresh, 1, 0)
    thresh = cv.convertScaleAbs(thresh)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) != 0:

        #find the biggest area of the contour
        c = max(contours, key = cv.contourArea)

        if original_img_ is not None:
            # the contours are drawn here
            cv.drawContours(original_img_, [c], -1, 255, 3)

        x,y,w,h = cv.boundingRect(c)
        print(x, y, w, h)
        return x, y, w, h