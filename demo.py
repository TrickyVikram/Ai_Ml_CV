
import cv2

#read imge

image=cv2.imread("/Users/vikramkumar/Desktop/Ai_Ml_project/vikram.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




