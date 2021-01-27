import numpy as np
from PIL import Image
import numpy.linalg
import imageio
import time
import matplotlib as plt
import matplotlib.pyplot as plt
from tkinter import *
def SVD(B, k):
    start_time = time.time()
    U, Sigma, V = np.linalg.svd(B.copy())
    print("*** %s Timpul de executie al programului in secunde este ***" % (time.time() - start_time))
    compressed = np.matrix(U[:,:k]) * np.diag(Sigma[:k]) * np.matrix(V[:k,:])
    return U,Sigma,V

def SVD2 (B,k):
    U, Sigma, V = np.linalg.svd(B.copy())
   # compressed = np.matrix(U[:,:k]) * np.diag(Sigma[:k]) * np.matrix(V[:k,:])
    plt.semilogy(Sigma)
    plt.show()
    sigmas = np.diag(Sigma)
    plt.plot(np.cumsum(Sigma) / np.sum(Sigma))
    plt.show()
    return U,Sigma,V


def grafice(image,k):

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    ur, sr, vr = SVD2(r, k)
    ug, sg, vg = SVD2(g, k)
    ub, sb, vb = SVD2(b, k)

def show_images(image_name):
   
    plt.title("Original_image.jpg")
    plt.imshow(image_name)
    plt.axis('on')
    plt.show()


    
def compress_image(image, k):

    original_bytes = image.nbytes
    print("Spatiul(in bytes) pentru a stoca aceasta imagine este = ", original_bytes)
    image = image/255
    linie,coloana,_= image.shape
    #vom imparti matricea in 3 matrice 2D
    Red = image[:, :, 0]
    Green = image[:, :, 1]
    Blue = image[:, :, 2]
    #print(Red)
    #print(Green)
    #print(Blue)

    U_Red, Sigma_Red, V_Red = SVD(Red, k)
    U_Green, Sigma_Green, V_Green =SVD(Green, k)
    U_Blue, Sigma_Blue, V_Blue = SVD(Blue, k)
  #  print(U_Red + "* "+Sigma_Red+ "*" +V_Red)

    bytes_matrices =sum([matrix.nbytes for matrix in [U_Red, Sigma_Red, V_Red, U_Green, Sigma_Green,
                                                       V_Green ,U_Blue, Sigma_Blue, V_Blue]])
    print("Matricile pe care le stocăm au dimensiunea totală (în bytes)",bytes_matrices)
   
    U_red_k= U_Red[:,0:k]
    V_red_k= V_Red[0:k,:]
    U_green_k= U_Green[:,0:k]
    V_green_k= V_Green[0:k,:]
    U_blue_k= U_Blue[:,0:k]
    V_blue_k= V_Blue[0:k,:]
    Sigma_Red_k = Sigma_Red[0:k]
    Sigma_Green_k= Sigma_Green[0:k]
    Sigma_Blue_k= Sigma_Blue[0:k]

    compressedBytes =sum([matrix.nbytes for matrix in [U_red_k, Sigma_Red_k, V_red_k, U_green_k, 
                                                       Sigma_Green_k, V_green_k ,U_blue_k, Sigma_Blue_k, V_blue_k]])
    print("Matricele compresate pe care vrem sa le stocam sunt ",compressedBytes)
    
    rata =100 * compressedBytes/original_bytes
    print("Rata de compresie dintre dimensiunea totala a imaginei originale si a factorilor comprimati este"  ,rata,"% din original")
   
    image_red_aproximation = np.matrix(U_Red[:, :k]) * np.diag(Sigma_Red[:k])* np.matrix(V_Red[:k, :])
    image_green_aproximation = np.matrix(U_Green[:, :k]) * np.diag(Sigma_Green[:k]) * np.matrix(V_Green[:k, :])
    image_blue_aproximation = np.matrix(U_Blue[:, :k]) * np.diag(Sigma_Blue[:k])* np.matrix(V_Blue[:k, :])
    

    compressed_image = np.zeros((linie,coloana,3))
    
  
    compressed_image[:, :, 0] = image_red_aproximation
    compressed_image[:, :, 1] = image_green_aproximation
    compressed_image[:, :, 2] = image_blue_aproximation


    np.clip(compressed_image,0,255,out=compressed_image)


    plt.title("Image Name: ")
    plt.imshow(compressed_image)
    plt.axis('off')
    #plt.show()
    return compressed_image

def menu():
   
    path = input("Please select an image= ")
    image = imageio.imread(path)
    print("*****Welcome to Compress Images*****")
    print()
    
    choice = input("""
                      1: Show the Image
                      2: Red color image
                      3: Green color image
                      4: Blue color image 
                      5: Compress the RGB image
                      6: Show Graphics
                      7: Compress the image with 4 different values
                      8: Adjust Image Brightness
                      9: Exit

                   Please enter your choice: """)

    if choice == "1":
       image=image/255
       rand,coloana,_=image.shape
       print("Pixels ",rand,"X",coloana)
       show_images(image)
    elif choice == "2":
       path = input("Please select an image= ")
       image=Image.open(path)
       red_band =image.getdata(band=0)
       img_mat = np.array(list(red_band), float) 
       img_mat.shape = (image.size[1], image.size[0])
       img_mat = np.matrix(img_mat)
       plt.title("Red_Band_image.jpg")
       plt.imshow(img_mat)
       plt.axis('on')
       plt.show()
    elif choice == "3":
       path = input("Please select an image= ")
       image=Image.open(path)
       green_band =image.getdata(band=1)
       img_mat = np.array(list(green_band), float) 
       img_mat.shape = (image.size[1], image.size[0])
       img_mat = np.matrix(img_mat)
       plt.title("Green_Band_image.jpg")
       plt.imshow(img_mat)
       plt.axis('on')
       plt.show()
    elif choice == "4":
       path = input("Please select an image= ")
       image=Image.open(path)
       blue_band =image.getdata(band=2)
       img_mat = np.array(list(blue_band), float) 
       img_mat.shape = (image.size[1], image.size[0])
       img_mat = np.matrix(img_mat)
       plt.title("Blue_Band_image.jpg")
       plt.imshow(img_mat)
       plt.axis('on')
       plt.show()
    elif choice == "5":
      rand,coloana,_=image.shape
      print("Pixels ",rand,"X",coloana)
      k = int(input("Introduceti valoarea lui k= "))
      image10 = compress_image(image, k)
      plt.title("Compressed_image.jpg")
      plt.imshow(image10)
      plt.axis('on')
      plt.show()
    elif choice == "6":
       k = int(input("Introduceti valoarea lui k= "))
       grafice(image,k)
    elif choice == "7":
       image1=compress_image(image,5)
       image2=compress_image(image,20)
       image3=compress_image(image,50)
       image4=compress_image(image,100)
       fig, axs = plt.subplots(2, 2,figsize=(7,7))
       axs[0, 0].imshow(image1)
       axs[0, 0].set_title('Compress Image: k= 5 ', size=10)
       axs[0, 1].imshow(image2)
       axs[0, 1].set_title('Compress Image: k=20', size=10)
       axs[1, 0].imshow(image3)
       axs[1, 0].set_title('Compress Image: k=50', size=10)
       axs[1, 1].imshow(image4)
       axs[1, 1].set_title('Compress Image: k=100', size=10)
       plt.tight_layout()
       plt.savefig('reconstructed_images_using_different_values.jpg',dpi=150)
       plt.show()

    elif choice=="8":
       path=input("Please select an image= ")
       im = Image.open(path)
       enhancer = ImageEnhance.Brightness(im)
       factor = 1 
       im_output = enhancer.enhance(factor)
       im_output.save('original-image.png')
       plt.imshow(im_output)
       plt.show()
       factor =0.3
       im_output = enhancer.enhance(factor)
       im_output.save('darkened-image.png')
       plt.imshow(im_output)
       plt.show()
       factor =2.8
       im_output = enhancer.enhance(factor)
       plt.imshow(im_output)
       plt.show()
    elif choice=="8":
        exit()
    else:
        print("Please try again")
        menu()

def main():
    menu()

main()
#image = imageio.imread(path)
#compressed=compress(image,32,0)
#show_images(compressed)
#print("*** %s Timpul de executie al programului in secunde este ***" % (time.time() - start_time))
