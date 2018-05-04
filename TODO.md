## MAIN TODO
* figure out pixel weight map-> Can't until I get labels

> set up weight initalization according to paper
    * well, the purpose of weight initalization is to preserve the variance of the normalized input. So should I be normalize the dataset?

> normalize each pixel
> test Standarize class
> somehow down_size before fitting

* label images with Ilastik
* code up Gaussian weight map


## Fake Data TODO
Purpose of this is to 
1. practie tuning hyperparameters
2. test that neural network is outputting results

* Tell Sebastian we can't increase the kernel size without drastically decreasing the output image

> figure out diameter of average cell: 30
* add layer parameter and see what it does to data
* code up training protocol
    * gaussian convolve fake data
    * defin accuracy function once I understand input output disparity
    * figure out class weight map per imag: new optimizer after every iteration to account for weight map
    * figure out what needs to be done differently with test set.

* figure out why vim autocomplete is so slow now


## Questions
* Yeah, so average cell has a diameter of roughly 30

## Completed
* Figure out whether I should be using transpose or UpSample
    * Just go with tranpose because according to this guy on stackexchane "In segmentation, we first downsample the image to get the features and then upsample the image to generate the segments."
* Figure out how to crop image and perserve the computational graph
* complete network
* Understand how to use 2D CrossEntropyLoss
    * does reshaping preserve differentiability?Doesn't matter, as I am using 2D Cross Entropy Loss
* Why am I uncomfortable with cropping?
    * In some senses, it should help because you feed the network useless information. 
    * You are teaching the neural network to not care about the boundary.


-1. Test on original dataset
0. learn how overlay simple cell image with a probability distibution using Monte Carlo
1. cell/vs no cell
2. multiple cells-close together

* show output of one cell
