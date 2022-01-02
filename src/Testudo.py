import numpy as np
import cv2
import copy

variable = int(input("Enter your video selection: "))
if variable == 1:
    cap = cv2.VideoCapture('../media/input/Tag0.mp4')
elif variable == 2:
    cap = cv2.VideoCapture('../media/input/Tag1.mp4')
elif variable == 3:
    cap = cv2.VideoCapture('../media/input/Tag2.mp4')
elif variable == 4:
    cap = cv2.VideoCapture('../media/input/multipleTags.mp4')
else:
    print("Incorrect input, restart code")
    exit()

testudo = cv2.imread('../media/input/testudo.png')
testudo_resize = cv2.resize(testudo, (200, 200))

dimension = 200
pos = np.array([
    [0, 0],
    [dimension - 1, 0],
    [dimension - 1, dimension - 1],
    [0, dimension - 1]], dtype="float32")

# Decode the tag ID and 4 digit binary and orientation
def decode(image):
    _, bwimg = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    corner = 255
    crop = bwimg[50:150, 50:150]

    data1 = crop[37, 37]
    data3 = crop[62, 37]
    data2 = crop[37, 62]
    data4 = crop[62, 62]
    white = 255
    if data3 == white:
        data3 = 1
    else:
        data3 = 0
    if data4 == white:
        data4 = 1
    else:
        data4 = 0
    if data2 == white:
        data2 = 1
    else:
        data2 = 0
    if data1 == white:
        data1 = 1
    else:
        data1 = 0

    if crop[85, 85] == corner:
        return list([data3, data4, data2, data1]), "Bottom Right"
    elif crop[15, 85] == corner:
        return list([data4, data2, data1, data3]), "Top Right"
    elif crop[15, 15] == corner:
        return list([data2, data1, data3, data4]), "Top Left"
    elif crop[85, 15] == corner:
        return list([data1, data3, data4, data2]), "Bottom Left"

    return None, None


# Function to return the order of points in camera frame

def order(points):
    box = np.zeros((4, 2), dtype="float32")

    sum = points.sum(axis=1)
    box[0] = points[np.argmin(sum)]
    box[2] = points[np.argmax(sum)]

    difference = np.diff(points, axis=1)
    box[1] = points[np.argmin(difference)]
    box[3] = points[np.argmax(difference)]

    # return the ordered coordinates
    return box


# Function to compute homography between world and camera frame
def hom(p0, pos):
    A = []
    pos2 = order(p0)

    for i in range(0, len(pos)):
        x, y = pos[i][0], pos[i][1]
        u, v = pos2[i][0], pos2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1, :] / Vh[-1, -1]
    h = np.reshape(l, (3, 3))
    return h


def warp(homo, img, hmax, wmax):
    inv = np.linalg.inv(homo)
    warp1 = np.zeros((hmax, wmax, 3), np.uint8)
    for h in range(hmax):
        for w in range(wmax):
            filler = [h, w, 1]
            filler = np.reshape(filler, (3, 1))
            i, j, k = np.matmul(inv, filler)
            warp1[h][w] = img[int(j / k)][int(i / k)]
    return (warp1)

# Generate contours to detect corners of the tag
def Contour(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edge = cv2.Canny(blur, 75, 200)
    edge_ = copy.copy(edge)
    clist = []

    _, contrs, hier = cv2.findContours(edge_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    node = []
    for h in hier[0]:
        if h[3] != -1:
            node.append(h[3])

    # loop over the contours
    for i in node:
        perimeter = cv2.arcLength(contrs[i], True)
        cont_approx = cv2.approxPolyDP(contrs[i], 0.02 * perimeter, True)

        if len(cont_approx) > 4:
            perimeter_ = cv2.arcLength(contrs[i - 1], True)
            corners = cv2.approxPolyDP(contrs[i - 1], 0.02 * perimeter_, True)
            clist.append(corners)

    new_clist = []
    for contour in clist:
        if len(contour) == 4:
            new_clist.append(contour)
    final_clist = []
    for elem in new_clist:
        if cv2.contourArea(elem) < 2500:
            final_clist.append(elem)

    return final_clist


# Reorient the tag based on the original orientation
def change_orientation(loc, dmax):
    if loc == "Bottom Right":
        pos = np.array([
            [0, 0],
            [dmax - 1, 0],
            [dmax - 1, dmax - 1],
            [0, dmax - 1]], dtype="float32")
        return pos
    elif loc == "Top Right":
        pos = np.array([
            [dmax - 1, 0],
            [dmax - 1, dmax - 1],
            [0, dmax - 1],
            [0, 0]], dtype="float32")
        return pos
    elif loc == "Top Left":
        pos = np.array([
            [dmax - 1, dmax - 1],
            [0, dmax - 1],
            [0, 0],
            [dmax - 1, 0]], dtype="float32")
        return pos

    elif loc == "Bottom Left":
        pos = np.array([
            [0, dmax - 1],
            [0, 0],
            [dmax - 1, 0],
            [dmax - 1, dmax - 1]], dtype="float32")
        return pos


# main function to process the tag
def process(frame, pos):
    retimg = []
    final_clist = Contour(frame)
    tlist = []

    for i in range(len(final_clist)):
        cv2.drawContours(frame, [final_clist[i]], -1, (0, 255, 0), 2)

        trez = final_clist[i][:, 0]
        H = hom(pos, order(trez))

        testudotag = cv2.warpPerspective(frame, H, (200, 200))

        cv2.imshow("Outline", frame)
        cv2.imshow("Tag after Homo", testudotag)

        testudotag_ = cv2.cvtColor(testudotag, cv2.COLOR_BGR2GRAY)
        dcod, loc = decode(testudotag_)
        null = np.full(frame.shape, 0, dtype='uint8')
        if not loc == None:
            pos2 = change_orientation(loc, 200)
            if not dcod == None:
                print("ID detected: " + str(dcod))
            homo_test = hom(order(trez), pos2)
            overlap = cv2.warpPerspective(testudo_resize, homo_test, (frame.shape[1], frame.shape[0]))
            if not np.array_equal(overlap, null):
                tlist.append(overlap.copy())

    mask = np.full(frame.shape, 0, dtype='uint8')
    if tlist != []:
        for testudo in tlist:
            temp = cv2.add(mask, testudo.copy())
            mask = temp

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        r, bin = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        invmask = cv2.bitwise_not(bin)

        _3dmask = frame.copy()
        _3dmask[:, :, 0] = invmask
        _3dmask[:, :, 1] = invmask
        _3dmask[:, :, 2] = invmask
        final_mask = cv2.bitwise_and(frame, _3dmask)
        image_final = cv2.add(final_mask, mask)
        cv2.imshow("Testudo", image_final)
        retimg.append(image_final)


    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()

    return retimg

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
FPS_val = 25
video_save = cv2.VideoWriter("../media/output/Testudo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), FPS_val, (width, height))

# Read the input video frame by frame
while cap.isOpened():

    __, frame = cap.read()
    if __ == False:
        break
    img_ = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    ret_image = process(img_, pos)
    for i in ret_image:
        video_save.write(i)

cap.release()