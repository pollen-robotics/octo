import h5py
import sys

root = h5py.File(sys.argv[1],"r", rdcc_nbytes=1024**2 * 2)


# print(a for a in root["observations/qpos"])
# exit()
# for i in range(len(root["action"])):
#     print(root["action"][i])

for i, image in enumerate(root["observations/images/cam_head"]):
    print(i)
    print(image)
    # cv2.imshow("im", im)
    # cv2.waitKey(1)
