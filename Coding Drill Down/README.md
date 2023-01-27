#Team members
1> Abhinav Rana (rabhinavcs@gmail.com)

2> Prashant Shinagare (techemerging1@gmail.com)

3> Yadunandan Huded (yadunandanlh64@gmail.com)

4> Pruthiraj Jayasingh (data.pruthiraj@gmail.com)

# **<u>Model Number</u>**

## **Target**

1> Achieve a Model Testing  accuracy of  99.4%  at least 4 times .

2> Have a limit of 10k parameters in model .

3>Have a limit of 15 Epochs .

# **<u>Model 1</u>**

## **Target **

1>Let's build the stricture of the model first.

2>Understanding the Statistics of Image and i baling the image.

3>Building model with Relu at every layer and Softmax as activation.  

2>Let's see the model performance for 20 epochs to understand where it is going.

## **Result**

```python
Total params: 51,450
Trainable params: 51,450
Non-trainable params: 0
```

Best Training Accuracy : Accuracy = 99.16

Best Testing Accuracy: Accuracy = 98.99%

## Analysis

Here  we are having very large number of parameters  but we have the structure ready. Mnist is a simple data set for number detection . We can build the model with less number of parameters as  we have simple features in this data set. First training accuracy is 11.30 % and testing accuracy 15.96%. Training accuracy is  not constant. Till epoch 13  model was good . After that it became over fit.

# **<u>Model 2-3</u>**

## **Target**

Reduce the number of parameters as much we can .

## **Result**

```
Total params: 7,690
Trainable params: 7,690
Non-trainable params: 0
```

Best Training Accuracy : Accuracy = 98.70%

Best Testing Accuracy: Accuracy =98.89%

## Analysis

Model is light now. We can see it is stable . It might get good accuracy at more epochs. But we need the result in less epochs. Now we ca-n try normalizing the input data by adding batch normalization.

# **Model 4**

## **Target**

Adding Batch Normalization  so that model can be generalise and get better accuracy .

## **Result**

```
Total params: 7,840
Trainable params: 7,840
Non-trainable params: 0
```

Best Training Accuracy : Accuracy = 99.57

Best Testing Accuracy: Accuracy = 99.61%

## Analysis

Well we got what we needed. But  In model's training accuracy we got above 99.00 after 5th  epochs. This model is good as we can see very less difference between training and testing accuracy.  But the testing  accuracy seems dropping at 10th epochs and  training  accuracy keeps on increasing. If we use dropout it will be tough for the model to learn and in that way we might be able to learn better.

# **Model 5**

## **Target**

Let's try Droupout in the model and see if the model is increasing it's performance and Stability .

## **Result**

```
Total params: 7,840
Trainable params: 7,840
Non-trainable params: 0
```

Best Training Accuracy : Accuracy = 99.27

Best Testing Accuracy: Accuracy =99.68%

## Analysis

After tuning the Model with batch size 64 and learning rate  0.015 we were able to achieve 99.68 . Dropout gave more stable model. 

