# Neural Architecture

 Building architecture is interesting to learn. Investigating what is unfolding within is more interesting.

- Why do we add layers?

If we consider the level of detail in an image. We may imagine that an item in a picture could be built up from smaller components. These little things can be made from various textures and patterns. Edges and gradients can be used to create textures and patterns.

To achieve this mechanically, we create layers. Our initial layers should be able to extract basic features like edges and gradients, as expected. Then, slightly sophisticated characteristics like textures and patterns would be built in the next layers. Subsequently layers might then construct object pieces that could later be assembled into whole objects. 


- Receptive Fields.

  ![Response form CharGPT ](/images/Screenshot from 2022-12-16 23-01-56.png)

For a Network Like 

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
            Conv2d-2            [-1, 8, 28, 28]             584
         MaxPool2d-3            [-1, 8, 14, 14]               0
            Conv2d-4           [-1, 16, 14, 14]           1,168
            Conv2d-5           [-1, 16, 14, 14]           2,320
         MaxPool2d-6             [-1, 16, 7, 7]               0
            Conv2d-7             [-1, 32, 5, 5]           4,640
            Conv2d-8             [-1, 64, 3, 3]          18,496
            Conv2d-9             [-1, 10, 1, 1]           5,770
================================================================
Total params: 33,058
Trainable params: 33,058
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.17
Params size (MB): 0.13
Estimated Total Size (MB): 0.30
----------------------------------------------------------------
```

Receptive field Looks like 

```python
=======================================Reciptive Field Calculator========================================
|    | Kernel_size   | Padding   |   Stride | Input_Img_size   | Output_Img_size   | Receptive_field   |
|---:|:--------------|:----------|---------:|:-----------------|:------------------|:------------------|
|  0 | 3*3           | 1         |        1 | 28*28            | 28*28             | 3*3               |
|  1 | 3*3           | 1         |        1 | 28*28            | 28*28             | 5*5               |
|  2 | 2*2           | NO        |        2 | 28*28            | 14*14             | 6*6               |
|  3 | 3*3           | 1         |        1 | 14*14            | 14*14             | 10*10             |
|  4 | 3*3           | 1         |        1 | 14*14            | 14*14             | 14*14             |
|  5 | 2*2           | NO        |        2 | 14*14            | 7*7               | 16*16             |
|  6 | 3*3           | NO        |        1 | 7*7              | 5*5               | 24*24             |
|  7 | 3*3           | NO        |        1 | 5*5              | 3*3               | 32*32             |
|  8 | 3*3           | NO        |        1 | 3*3              | 1*1               | 40*40             |
=========================================================================================================
```


- Convolution Mathematics

   [Math_Convgif](images/Math_Convgif) 

  

- Kernels in Layers . 

  If we look at 3*3 Kernels in a Conv Layers after trainning itr might look like bellow.

  

  ![kernel.png](/images/kernel.png)

  

  The outcome of the initial Layer of Convolution may look like this for MNIST.

![after_conv]()

