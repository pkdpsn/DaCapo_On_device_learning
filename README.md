# DaCapo On Device Learning

We have tested our code on Pycharm running on Windows 11.

## FINDING OPTIMAL SEGMENTATION FOR THE MODEL
1. Import the model file 
```modelXC_2F```
into the Segmentation_finder file.

2. Set the amount of available SRAM in the variable M.

3. Run the code to find the optimal segmentation

## PRODUCING RESULTS
1. Import the model file 
```modelXC_2F / modelXC_2Fcheckpointing```
into the file. The optimal checkpoints for this model have already been calculated forward pass has been written for these model files. 

2. Run the code to produce the timing and ram consumption graphs to evaluate our results.

The results have been evaluated by using only 1 core and setting utilization limit to 10% as the original code was meant to be run on STM32 microcontroller.