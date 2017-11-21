# Neural_Network
This time 3 NN are built: dense NN from scratch, vgg-16, resnet.

Because the limitation of computation ability, only dense NN and resnet are trained.

It turns out resnet outperforms dense NN. Dense NN could only get 56% of accuracy without batch norm in CIFAR10, while resnet 
could reach 86%, and it has only 14 conv layers. You can see the result in 'train_scratch.ipynb' and 'train_resNet2.ipynb'.

It is exciting to using image classifier in robot, simply retrain the few last layers, and deploy it in the server.py.
see https://www.youtube.com/watch?v=gszTsfhCNMg&feature=youtu.be

It is interesting to visulize the weights in different conv layers. It turns out that, weights in conv layers tend to be Gaussian distribution while weights in dense layers is pretty random.

See the accuracy and lost:
![alt text](https://github.com/zyxstudycs/Neural_Network/blob/master/images/loss_accuracy.png)

The weights in conv layers:
![alt text](https://github.com/zyxstudycs/Neural_Network/blob/master/images/conv1.png)
![alt text](https://github.com/zyxstudycs/Neural_Network/blob/master/images/conv2.png)

The weights in dense layer:
![alt text](https://github.com/zyxstudycs/Neural_Network/blob/master/images/dense.png)
