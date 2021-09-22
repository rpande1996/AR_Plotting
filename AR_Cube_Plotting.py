import numpy as np
import cv2
import copy
from numpy.linalg import inv
from numpy.linalg import norm
from itertools import product

cap = cv2.VideoCapture('Tag0.mp4')
#cap = cv2.VideoCapture('Tag1.mp4')
#cap = cv2.VideoCapture('Tag2.mp4')
#cap = cv2.VideoCapture('multipleTags.mp4')

dimension = 200
pos = np.array([
    [0, 0],
    [dimension - 1, 0],
    [dimension - 1, dimension - 1],
    [0, dimension - 1]], dtype="float32")


# Decode the tag ID and 4 digit binary and orientation
def decode(image):
    ret, img_bw = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    cropped_img = img_bw[50:150, 50:150]

    center_locs = [37, 62]
    corner_locs = [15, 85]
    code = {}
    rotation_dict = {"BL": 2, "TL": 1, "TR": 0, "BR": 3}
    for x, y in product(center_locs, center_locs):
        block = cropped_img[x, y]
        if block == 255:
            code[str(x) + '_' + str(y)] = 1
        else:
            code[str(x) + '_' + str(y)] = 0

    order = list(product(center_locs, center_locs))
    order = [str(item[0]) + '_' + str(item[1]) for item in order]
    code_list = [code[loc] for loc in order]
    code_list[2], code_list[3] = code_list[3], code_list[2]

    orientation = []
    for x, y in product(corner_locs, corner_locs):
        if cropped_img[x, y] == 255:
            if y == 15:
                orientation.append("T")
            else:
                orientation.append("B")
            if x == 15:
                orientation.append("L")
            else:
                orientation.append("R")
            orientation = "".join(orientation)
            rotation = rotation_dict[orientation]
            code_list = code_list[rotation:] + code_list[:rotation]
            return code_list, orientation

    return None, None


def Cube(image, image_points):  # To draw the cube
    image_points = np.int32(image_points).reshape(-1, 2)
    # draw ground floor in green
    image = cv2.drawContours(image, [image_points[:4]], -1, (0, 255, 255), 3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        image = cv2.line(image, tuple(image_points[i]), tuple(image_points[j]), (255, 255, 0), 3)

    # draw top layer in red color
    image = cv2.drawContours(image, [image_points[4:]], -1, (0, 0, 255), 3)
    return image


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

# Function to calculate Projection matrix
def projectionmat(h):
    K = np.array(
        [[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]]).T
    h = inv(h)
    b_new = np.dot(inv(K), h)
    b1 = b_new[:, 0].reshape(3, 1)
    b2 = b_new[:, 1].reshape(3, 1)
    r3 = np.cross(b_new[:, 0], b_new[:, 1])
    b3 = b_new[:, 2].reshape(3, 1)
    L = 2 / (norm((inv(K)).dot(b1)) + norm((inv(K)).dot(b2)))
    r1 = L * b1
    r2 = L * b2
    r3 = (r3 * L * L).reshape(3, 1)
    t = L * b3
    r = np.concatenate((r1, r2, r3), axis=1)

    return r, t, K


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


def process(frame, pos):
    retcube = []
    final_clist = Contour(frame)
    clist = []
    axis = np.float32(
        [[0, 0, 0], [0, 200, 0], [200, 200, 0], [200, 0, 0], [0, 0, -200], [0, 200, -200], [200, 200, -200],
         [200, 0, -200]])
    mask = np.full(frame.shape, 0, dtype='uint8')
    for i in range(len(final_clist)):
        cv2.drawContours(frame, [final_clist[i]], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", frame)
        c_rez = final_clist[i][:, 0]
        H = hom(pos, order(c_rez))
        tag = cv2.warpPerspective(frame, H, (200, 200))

        cv2.imshow("Outline", frame)
        cv2.imshow("Tag after homogenous", tag)

        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        decoded, location = decode(tag1)
        if not location == None:
            if not decoded == None:
                r, t, K = projectionmat(H)
                points, jac = cv2.projectPoints(axis, r, t, K, np.zeros((1, 4)))
                img = Cube(mask, points)
                clist.append(img.copy())
    if clist != []:  # empty cube list
        for cube in clist:
            temp = cv2.add(mask, cube.copy())
            mask = temp

        final_image = cv2.add(frame, mask)
        cv2.imshow("Cubes", final_image)
        retcube.append(final_image)

    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()

    return retcube

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
FPS_val = 25
video_save = cv2.VideoWriter("AR_Cube.mp4", cv2.VideoWriter_fourcc(*'mp4v'), FPS_val, (width, height))

# Read the input video frame by frame
while cap.isOpened():
    success, frame = cap.read()
    if success == False:
        break
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    ret_image = process(img, pos)

    for image in ret_image:
        video_save.write(image)

cap.release()
