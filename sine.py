import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL.Image import Image
import numpy as np
import cv2
img = cv2.imread('pisaa1.png', cv2.IMREAD_UNCHANGED)
print('Original shape : ', img.shape)
H,W,_ =img.shape
dsize = (H,W)#dsize属性值第一个数对应列数，第二个数对应行数

fx = 2#列数变为原来的1.5倍
fy = 2#行数变为原来的1.5倍
resized1 = cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
#最邻近插值
resized2 = cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
#双线性插值

cv2.imwrite("INTER_LINEAR image.png", resized2)
cv2.imwrite("INTER_NEAREST.png", resized1)
img_test=[]
img1 = mpimg.imread('baboon1.png')
img2 = mpimg.imread('INTER_LINEAR image.png')
img3 = mpimg.imread('INTER_NEAREST.png')
img4 = mpimg.imread('rainbowwe12.png')
img1.reshape((4*H*W,3))
img2.reshape((4*H*W,3))
img3.reshape((4*H*W,3))
img4.reshape((4*H*W,3))
r1=0
r2=0
r3=0
for k in range(3):
    mu_s1 = img1[:,:, k]
    mu_s2 = img2[:,:, k]
    mu_s3 = img3[:,:, k]
    mu_s4 = img4[:,:, k]
    sqe1 = (mu_s1 - mu_s2) ** 2
    sqe2 = (mu_s1 - mu_s3) ** 2
    sqe3 = (mu_s1 - mu_s4) ** 2
    rmse1 = np.sqrt(sqe1.sum() / len(img1))
    rmse2 = np.sqrt(sqe2.sum() / len(img1))
    rmse3 = np.sqrt(sqe3.sum() / len(img1))

    r1+=rmse1
    r2+=rmse2
    r3+=rmse3

print('bilinear',r1)
print('nearest',r2)
print('MOMoGP',r3)
# img_test.append(img1)
# img_test.append(img2)
# img_test.append(img3)
# img_test.append(img4)
# imgplot1 = plt.imshow(img1)
# plt.plot(img1)
# plt.savefig("baboon"+".pdf")
# plt.plot(img2)
# plt.savefig("bilinear"+".pdf")
# plt.plot(img3)
# plt.savefig("nearest"+".pdf")
# plt.plot(img4)
# plt.savefig("MOMoGP_maxmin"+".pdf")
# # fig = plt.figure(figsize=(10, 10))
# # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(2, 2))
# # names=['Original','Bilinear','Nearest','SPGPN(RMSE:0.173)']
# # for i in range(1,5):
# #     ax = plt.subplot(2,2,i)
# #     # ax.set_title(names[i-1])
# #     plt.imshow(img_test[i-1])
# #     plt.xticks([])
# #     plt.yticks([])
# #
# # plt.savefig('image_maxmin.pdf')
# # # plt.show()