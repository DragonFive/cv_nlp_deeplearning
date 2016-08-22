import cv2

img = cv2.imread(r'E:\×ÊÁÏ\onedrive\code\test\image\lena.png'.decode('utf-8').encode('gbk'))
cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey (0)
cv2.destroyAllWindows()