import cv2 as cv

cam = cv.VideoCapture(0)

cv.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv.imshow("test", frame)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv.destroyAllWindows()

path = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\FAD\\ACV_Ses2\\opencv_frame_0.png'
path_mask = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\FAD\\ACV_Ses2\\mask1.jpeg'
img = cv.imread(path)
mask = cv.imread(path_mask)
mask_cp = mask.copy()

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

T = 128
mask[mask > T] = 1
mask_cp[mask_cp < T] = 0
mask_bw = mask * mask_cp
mask_bw = cv.cvtColor(mask_bw, cv.COLOR_BGR2GRAY)

masked = img_gray * mask_bw

im_v = cv.vconcat([img_gray, masked])
# im_h = cv.hconcat([img_gray, masked])
cv.imshow('original and masked image', im_v)

# cv.imshow('webcam image', img)
# cv.imshow('mask', mask)
# cv.imshow('black and white mask', masked)
cv.imwrite('masked_webcam2.jpg', im_v)
cv.waitKey(0)