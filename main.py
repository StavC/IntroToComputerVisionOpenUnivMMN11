
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    # Question 1,A

    GaussianMatrix = np.random.normal(3, 7, size=(50, 50))
    print(GaussianMatrix)
    plt.figure()
    plt.imshow(GaussianMatrix,interpolation='nearest')
    plt.show()
    plt.imshow(GaussianMatrix, cmap='gray')
    plt.colorbar(ticks=[0, 255])
    plt.show()

    # Question 1,B

    histogram=plt.hist(GaussianMatrix)
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

    cannyEdges1=cv2.Canny(image,150,200)
    cannyEdges2=cv2.Canny(image,200,300)
    cannyEdges3=cv2.Canny(image,50,100)

    fig = plt.figure(figsize=(8, 4))
    plt1 = fig.add_subplot(1, 3, 1)
    plt.imshow(cannyEdges1)
    plt.title('1')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt2 = fig.add_subplot(1, 3, 2)
    plt.imshow(cannyEdges2)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('2')

    plt2 = fig.add_subplot(1, 3, 3)
    plt.imshow(cannyEdges3)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title('3')

    plt.show()



if __name__ == '__main__':
    main()


