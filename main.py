
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plot_cov_ellipse
from scipy import stats


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
    # Question 2.A
    cov=[[2.48,0.94],[0.94,2.04]]
    print(cov)
    pos=[-1,2]


    ellip,x,y=plot_cov_ellipse.plot_cov_ellipse(cov,pos,nstd=1)

    # Question 2.B
    #points=np.random.multivariate_normal(pos,cov,10).T
    points=np.array([x,y])
    covMatrix=np.cov(points)

    #print(f' points:  {points} \n')
    pos2=np.mean(points, axis=1)
    print(f'Average {pos2}')
    print(covMatrix)
    dst=0.5*(np.trace(np.multiply(np.power(cov,-1),covMatrix))-np.log(np.linalg.det(covMatrix)/np.linalg.det(cov))) # Kullback–Leibler divergence from wikipedia
    print(f' the K-L Distance is: {dst}')
    dst2=np.linalg.norm(np.subtract(cov,covMatrix) + np.linalg.norm(np.subtract(pos,pos2))) #Frobenius norm
    print(f' the Frobenius Norm is : {dst2}')
    dst3 = np.power(np.linalg.norm(pos - pos2), 2) + np.power(np.linalg.norm(cov - covMatrix), 2)
    print(f' The Loss is : {dst3}')

    #Question 2.C




    a=np.linspace(-5,3,20)
    b=np.linspace(-2,6,20)
    points2=[]
    for i in range(20): #Building the Grid of points
        for j in range(20):
            points2.append([a[int(i)],b[int(j)]])

    plt.imshow(points2)
    plt.show()


    picture1=stats.multivariate_normal.pdf(points2,pos,cov)
    picture1=picture1.reshape(20,20)

    plt.imshow(picture1)
    plt.show()

    picture2=stats.multivariate_normal.pdf(points2,pos2,covMatrix)
    picture2=picture2.reshape(20,20) # 10 10
    plt.imshow(picture2)
    plt.show()

    plt.imshow(picture2-picture1)
    plt.show()
    plt.imshow(picture1-picture2)
    plt.show()


    #Question 2.D

    cov = [[2.48, 0.94], [0.94, 2.04]]
    pos=[-1,2]
    #K_L_Distances=[]
    frobenius=[]
    pointsArray=[]
    Loss=[]
    for i in range(50):
        num_points=10+i*5 #number of points to sample
        points=np.random.multivariate_normal(pos,cov,num_points).T
        covMatrix=np.cov(points)
        pos2 = np.mean(points, axis=1)
        #dst = 0.5 * (np.trace(np.multiply(np.power(cov, -1), covMatrix)) - np.log( np.linalg.det(covMatrix) / np.linalg.det(cov)))  # Kullback–Leibler divergence from wikipedia
        dst2 = np.linalg.norm(np.subtract(cov, covMatrix)) + np.linalg.norm(np.subtract(pos,pos2))
        dst3=np.power(np.linalg.norm(pos-pos2),2) + np.power(np.linalg.norm(cov-covMatrix),2)
        pointsArray.append(num_points)
        #K_L_Distances.append(dst)
        frobenius.append(dst2)
        Loss.append(dst3)
    print(pointsArray)
    #print(K_L_Distances)
    print(frobenius)
    print(Loss)

    plt.plot(pointsArray,Loss,label='Loss')
    plt.plot(pointsArray,frobenius,label='Frobenius')
    #plt.plot(pointsArray,K_L_Distances,label='K-L')
    plt.legend()
    plt.show()
    print(f' the min distance is : {min(Loss)} when sampling : {Loss.index(min(Loss))*5+10}')

def Question3():

    sun1 = cv2.imread('sun.jpeg')
    sun2 = cv2.imread('sun2.jpg')
    sun1=cv2.resize(sun1,(512,512))
    sun2=cv2.resize(sun2,(512,512))

    mix = np.hstack((sun1[:, :256], sun2[:, 256:]))

    # creating Gaussian Pyramids for both sun pictures
    sun1Copy = sun1.copy()
    GPyramidSun1 = [sun1Copy]
    for i in range(10):
        sun1Copy = cv2.pyrDown(sun1Copy)
        GPyramidSun1.append(sun1Copy)

    sun2Copy = sun2.copy()
    GPyramidsun2 = [sun2Copy]
    for i in range(10):
        sun2Copy = cv2.pyrDown(sun2Copy)
        GPyramidsun2.append(sun2Copy)

    #now creat laplacian pyramid for both sun pictures
    sun1Copy = GPyramidSun1[9]
    LapPyramidSun1 = [sun1Copy]
    for i in range(9, 0, -1):
        GausLayer = cv2.pyrUp(GPyramidSun1[i])
        lapLayer = cv2.subtract(GPyramidSun1[i - 1], GausLayer)
        LapPyramidSun1.append(lapLayer)

    sun2Copy = GPyramidsun2[9]
    LapPyramindSun2 = [sun2Copy]
    for i in range(9, 0, -1):
        GausLayer = cv2.pyrUp(GPyramidsun2[i])
        lapLayer = cv2.subtract(GPyramidsun2[i - 1], GausLayer)
        LapPyramindSun2.append(lapLayer)

    sunsets = []
    for sun1Lap, sun2Lap in zip(LapPyramidSun1, LapPyramindSun2):
        cols, rows, ch = sun1Lap.shape
        laplacian = np.hstack((sun1Lap[:, 0:int(cols / 1.8)], sun2Lap[:, int(cols / 1.8):]))
        sunsets.append(laplacian)


    blendedPicture = sunsets[0]
    for i in range(1, 10):
        blendedPicture = cv2.pyrUp(blendedPicture)
        blendedPicture = cv2.add(sunsets[i], blendedPicture)


    cv2.imshow("Sun1", sun1)
    cv2.imshow("Sun2", sun2)
    cv2.imshow("Both Pictures", mix)
    cv2.imshow("blendedPicture", blendedPicture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    #Question1()
    Question2()
    #Question3()

