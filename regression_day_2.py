"""we will copy what we did yesterday in terms of data processing and focus on regression exclusivly
1) Try out base regression model with different hyper parameters and what each one of them does
2) try xgboost regression and study the difference hyper parameters
3) figure out different evaluation techniques
"""

# starting out with learning different evaluation techniques

# we are gonna consider following error metrics and study about them :

# RMSE, MAE, R2, concordance correlation coefficient, residual plots, library : yardstick

# topics that i need to read before i am sure what these metrics are about :

# residual plot, deviation, bias, variance,DV with large variance

# let's start with population vairance vs sample variance

import numpy as np
import math

# we know deviation is difference between observed value and some other
# value usually mean or any chose measure of central tendency
# observed_value = 0
# mean = 0
# deviation = np.absolute(observed_value-mean)
# this is different from standard deviation


def find_sample_variance(data_points):
    data_mean = sum(data_points)/len(data_points)
    variance = sum([(i-data_mean)**2 for i in data_points])/(len(data_points)-1)
    return variance


# print(find_sample_variance([1,2,3,4]))
# let's compare it to numpy var
# print(np.var([1,2,3,4]))
# ^ this above results divides the sum of mean by n and not by n-1
# specifing ddof = 1 we get
# print(np.var([1,2,3,4],ddof=1))

# https://web.ma.utexas.edu/users/mks/M358KInstr/SampleSDPf.pdf this article shows why n-1
# is makes the variance unbiased
# NOTE : the standard deviation calculated as a square root of variance is
# not unbiased because square root is not a linear function

# https://www.math.ucdavis.edu/~anne/WQ2007/mat67-Lh-Linear_Maps.pdf tells us what is a linear function
# square root is not linear because square root of (x+y) is not equal to sqrt(x) + sqrt(y)

# now that we are done with the basic understanding of statistics involved in error metrics, let's move ahead

# closing note : was able to lear a bit about variance, standard deviation
