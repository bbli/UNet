## Thoughts
* larger kernel size means one weight parameters affects less inputs-> less compromising

## New Gameplan

* try decay_cosine
* try different optimizer(Adam overfits, SGD compromises too much)

* try with conv padding(my guess is that it should perform worse)
* data augmentation once train accuracy goes above 90
* change loss function to dice
* freeze down layers



## Finished
* check padding again(make sure output is greater)
change lookup_table to a dictionary
    * should be not be manual, but rather controlled by a layer parameter
    * or **just stick with three layers**
* rewrite crop to throw error if output size is not greater than label



## Training Gameplan

> Double learning rate discover for SGD
    * so optimal is around 4e-3
> Cyclical Learning Rate for SGD

> kernel_size =[6,7,8] -> figure out padding adjustment
> lr=[1.2e-2,8e-3,4e-3]
> feature_maps = 32
> downsize = 5

> Hyperparameter tune with Adam
    * yup Adam went back to choosing blank canvas
> log test images to tensorboard


> hyperparameter tune with 64 feature maps
> write up code that saves image after run.(For non hyperparameter tuning)
    * well as long as you save the model, no need to save image in script

> try with nesterov
    * gave best result at 0.72
> stop jumping around after a while and use Adam
    * terrible results again
* Try 4 Layers
    * I wil need to use overlap tile strategy because reduction is too drastic.
        * Good time to think about effects of padding
* Save images based on time
    * change reduceTo2D to not yet argmax arrays

* Dropoout

* implement bagging, since from looking at the tensoboard graph of accuracy, we hit some local maximums along the way
* Differential Learning Rates at End(what is the hierarchy?)
* I can figure out difference between classification and segmentation by looking at the output results from ilastik
    * it seems classfication gives two output maps, while segmentation gives one

## Notes
* Adam will cause the learning rate to be too low, and thus stops at the local min of uniform canvas.
* scheduler helps SGD not "smear" out

* 3e-4 barely lowers the loss function with SGD
* 1.2e-2 SGD with kernel size 7 gave me my highest at 70% test score so far.

* I may need to increase complexity as I cannot get above 80% train accuracy.

## MAIN TODO
> set up weight initalization according to paper
    * well, the purpose of weight initalization is to preserve the variance of the normalized input. So should I be normalize the dataset?

> normalize each pixel
> test Standarize class
> somehow down_size before fitting

> label images with Ilastik

> rewrite Paryhyale Dataset class
> rewrite Train.py

> fix parhyale labels
* put print statements wherever you can in Train.py to check code

## Fake Data TODO
Purpose of this is to 
1. practie tuning hyperparameters
2. test that neural network is outputting results

> figure out diameter of average cell: 30
> add layer parameter and see what it does to data
> redo calcuation of receptive field
    * field = 6k-2.
    * I think a kernel of size 6 is a good compromise for this fake dataset. In any case, I should probably tune this(4,5,6,7,8) when I train on actual dataset(note you will need to write a script that pads accordingly)
> figure out why upsample doesn't work
    * upsample does not reduce the feature dimension


* code up training protocol
    > gaussian convolve fake data
    > pad fake data
    > rework tensor to fit with 3.0 CrossEntropyLoss
        * Nvm, looks like it has been backported
    > defin accuracy function once I understand input output disparity
    > figure out class weight map per for dataset
        * I should either change weight map in loop or
        * do it across the entire dataset once
    > figure out what needs to be done differently with test set.
        * no fit, just transform
* create different fake data to test empty prediction



## Thoughts
* Yeah, so average cell has a diameter of roughly 30
* Understand how to use 2D CrossEntropyLoss
    * does reshaping preserve differentiability?Doesn't matter, as I am using 2D Cross Entropy Loss
* Why am I uncomfortable with cropping?
    * In some senses, it should help because you feed the network useless information. 
    * You are teaching the neural network to not care about the boundary.
* Yeah, weight map per image should be better as it will punish misclassificaition on images with few cells
    * Nvm, weight map is best determined at beginning b/c padding will screw it up
    * Also, creating a new optimizer will screw up with the optimizer's state
* remember that index 0 of the feature channel is the background since you have a black blackground

## Completed
* Figure out whether I should be using transpose or UpSample
    * Just go with tranpose because according to this guy on stackexchane "In segmentation, we first downsample the image to get the features and then upsample the image to generate the segments."
* Figure out how to crop image and perserve the computational graph
* complete network


-1. Test on original dataset
0. learn how overlay simple cell image with a probability distibution using Monte Carlo
1. cell/vs no cell
2. multiple cells-close together

* show output of one cell

## ISBIDataset
* fix path
* check that imageToTorch and labelToTorch transfer over
* find image resolution and then find padding for kernel=3
