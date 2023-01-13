```
Requirement already satisfied: torchsummary in /home/jd/anaconda3/lib/python3.7/site-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
       BatchNorm2d-2            [-1, 8, 26, 26]              16
            Conv2d-3            [-1, 8, 24, 24]             584
       BatchNorm2d-4            [-1, 8, 24, 24]              16
            Conv2d-5           [-1, 16, 22, 22]           1,168
       BatchNorm2d-6           [-1, 16, 22, 22]              32
         MaxPool2d-7           [-1, 16, 11, 11]               0
            Conv2d-8             [-1, 16, 9, 9]           2,320
       BatchNorm2d-9             [-1, 16, 9, 9]              32
           Conv2d-10             [-1, 32, 7, 7]           4,640
      BatchNorm2d-11             [-1, 32, 7, 7]              64
           Conv2d-12             [-1, 16, 7, 7]             528
      BatchNorm2d-13             [-1, 16, 7, 7]              32
           Conv2d-14             [-1, 16, 5, 5]           2,320
      BatchNorm2d-15             [-1, 16, 5, 5]              32
           Conv2d-16             [-1, 32, 3, 3]           4,640
      BatchNorm2d-17             [-1, 32, 3, 3]              64
           Conv2d-18             [-1, 10, 1, 1]           2,890
================================================================
Total params: 19,458
Trainable params: 19,458
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.35
Params size (MB): 0.07
Estimated Total Size (MB): 0.43
----------------------------------------------------------------
```

# **My intuition behind the Code.**

1> we had to use < 20k parameters .

2> Had to use <20 Epochs

If i multiply **3x3x32x64**  we will have **18432**  gone already . so i didn't consider using a 64 channel at convolution. Also i   can't use  3x3x16x32 more.

I have used the batch size 64 ,128,256 and stared with learning rate 0.035. Gradually seeing the result at each epochs i increase the Learning rate. finally Got the sweet sport at **256 batch** size and **learning rate as 0.03772**. 

**<u>Observation (Might be wrong )</u>** 

​	I noticed  every time i  run a  model with different architecture ,if it reached a validation accuracy of  > 99.10 % at initial 3 epochs  then there is a scope for the  model to reach  >= 99.4 % validation accuracy.

​	Also i have Noticed if i use dropout in this network the model is more stable, but accuracy couldn't reach 99.4%  

If  I run the same model second after restarting the kernel, it didn't give the the max accuracy i got before. (result might vary) 

Below is the plot of accuracy and loss of second time run .

![image-20200815190757636](/home/jd/.config/Typora/typora-user-images/image-20200815190757636.png)



## **![image-20200815191459189](/home/jd/.config/Typora/typora-user-images/image-20200815191459189.png)Summary** 

​	I have Used 9 Convolution layers  and 8 BatchNorm2d layers after all Convolution layers except the last Convolution(conv9 ) layer. Also i have used Maxpooling after 3rd convolution layer.
I used the batch size of 256. I have started with 0.035 learning rate and got 99.43 accuracy at 0.0356 .Then to increase the accuracy i used 0.3772 .Also used 1x1 kernels once as conv6. All batch normalisation used after Relu and before  max pooling.  Total Number of perimeters used in this model is 19458.

![image-20200815184702710](/home/jd/.config/Typora/typora-user-images/image-20200815184702710.png)

```
Epochs 19
loss=0.001337677240371704 batch_id=234: 100%|██████████| 235/235 [00:08<00:00, 27.57it/s]  
  0%|          | 0/235 [00:00<?, ?it/s]
Test set: Average loss: 0.0210, Accuracy: 9948/10000 (99%)
```

Got the requires testing accuracy at 11,13,16,19 and 20th number epochs.



## **Try 2**

​			I have also tried the below network which was able to give me 99.43% validation accuracy 

```
Requirement already satisfied: torchsummary in /home/jd/anaconda3/lib/python3.7/site-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
         MaxPool2d-3            [-1, 8, 14, 14]               0
            Conv2d-4           [-1, 16, 12, 12]           1,168
       BatchNorm2d-5           [-1, 16, 12, 12]              32
            Conv2d-6           [-1, 32, 10, 10]           4,640
       BatchNorm2d-7           [-1, 32, 10, 10]              64
         MaxPool2d-8             [-1, 32, 5, 5]               0
            Conv2d-9             [-1, 32, 3, 3]           9,248
           Conv2d-10             [-1, 10, 1, 1]           2,890
================================================================
Total params: 18,138
Trainable params: 18,138
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.20
Params size (MB): 0.07
Estimated Total Size (MB): 0.27
```