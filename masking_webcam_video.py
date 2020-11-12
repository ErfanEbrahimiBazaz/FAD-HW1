import cv2 as cv


vid = cv.VideoCapture(0)
path_mask = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\FAD\\ACV_Ses2\\mask1.jpeg'
mask = cv.imread(path_mask)
mask_cp = mask.copy()

T = 128
mask[mask > T] = 1
mask_cp[mask_cp < T] = 0
mask_bw = mask * mask_cp
mask_bw = cv.cvtColor(mask_bw, cv.COLOR_BGR2GRAY)

while (True):
    ret, frame = vid.read()
    cv.imshow('frame', frame)
    b, g, r = cv.split(frame)
    cv.imshow('blue', b)
    cv.imshow('green', g)
    cv.imshow('red', r)

    masked_frame_b = b * mask_bw
    masked_frame_g = g * mask_bw
    masked_frame_r = r * mask_bw
    cv.imshow('masked blue', masked_frame_b)

    constructed_masked_image = cv.merge((masked_frame_b, masked_frame_g, masked_frame_r))
    cv.imshow('constructed masked image', constructed_masked_image)

    constructed_frame = cv.merge((b, g, r))
    cv.imshow('constructed frame', constructed_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()