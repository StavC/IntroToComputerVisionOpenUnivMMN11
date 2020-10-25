
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


def Question1():
    # Question 1,A

    GaussianMatrix = np.random.normal(3, 7, size=(50, 50))
    print(GaussianMatrix)
    plt.figure()
    plt.imshow(GaussianMatrix,interpolation='nearest')
    plt.show()
    plt.imshow(GaussianMatrix, cmap='gray') # displaying the matrix in gray scale
    plt.colorbar(ticks=[0, 255])
    plt.show()

    # Question 1,B

    hist=sns.distplot(GaussianMatrix) #plotting the historgram
    plt.show()

    # Question 1,C

    image=cv2.imread('Fitz.jpg')
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    imageGray=image.copy()
    imageGray=cv2.cvtColor(imageGray,cv2.COLOR_BGR2GRAY)

    fig = plt.figure(figsize=(8, 4))
    plt1 = fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('RGB')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt2 = fig.add_subplot(1, 2, 2)
    plt.imshow(imageGray,cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('Gray')
    plt.show()

    # Question 1 ,D

    cannyEdges1=cv2.Canny(image,150,250)
    cannyEdges2=cv2.Canny(image,175,250) #200 300
    cannyEdges3=cv2.Canny(image,200,250)

    sobelx = cv2.Sobel(imageGray, cv2.CV_8U, 1, 0, ksize=5)
    sobely = cv2.Sobel(imageGray, cv2.CV_8U, 0, 1, ksize=5)

    fig = plt.figure(figsize=(10, 10))
    plt1 = fig.add_subplot(2, 3, 1)
    plt.imshow(cannyEdges1)
    plt.title('150,250')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt2 = fig.add_subplot(2, 3, 2)
    plt.imshow(cannyEdges2)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('175,250')

    plt3 = fig.add_subplot(2, 3, 3)
    plt.imshow(cannyEdges3)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('200,250')

    plt4=fig.add_subplot(2,3,4)
    plt.imshow(sobelx)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('SobelX')

    plt5 = fig.add_subplot(2, 3, 5)
    plt.imshow(sobely)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('SobelY')

    plt6 = fig.add_subplot(2, 3, 6)
    sobelx = cv2.Sobel(imageGray, cv2.CV_64F, 1, 0, ksize=31)
    sobely = cv2.Sobel(imageGray, cv2.CV_64F, 0, 1, ksize=31)

    plt.imshow(np.sqrt(sobelx*sobelx + sobely*sobely))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('Mixed')

    plt.show()

    #Question 1E you can use both photos that i reviewed in the PDF ('shapes.jpg','Fitz.jpg')
    img=cv2.imread('Fitz.jpg')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=np.float32(gray)
    harris = cv2.cornerHarris(gray, 2,3, 0.04)
    harris = cv2.dilate(harris, None)
    img[harris > 0.01 * harris.max()] = [0, 0, 255]
    plt.imshow(img,cmap='gray')
    plt.show()


def Question2():
    print(' ')




if __name__ == '__main__':
    Question1()


