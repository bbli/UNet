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
3. practice data augmentation

* Tell Sebastian we can't increase the kernel size without drastically decreasing the output image

> create 11 train, 1 test and save it
* figure out why vim autocomplete is so slow now
* do base training on it
    * figure out class weight map per image
    * code up training protocol
        1. new optimizer after every iteration

* learn how to use torchsample
* figure out input ouput disparity!!!
* Define accuracy function


## Questions


## Completed
* Figure out whether I should be using transpose or UpSample
    * Just go with tranpose because according to this guy on stackexchane "In segmentation, we first downsample the image to get the features and then upsample the image to generate the segments."
* Figure out how to crop image and perserve the computational graph
* complete network
* Understand how to use 2D CrossEntropyLoss
    * does reshaping preserve differentiability?Doesn't matter, as I am using 2D Cross Entropy Loss


-1. Test on original dataset
0. learn how overlay simple cell image with a probability distibution using Monte Carlo
1. cell/vs no cell
2. multiple cells-close together

* show output of one cell
