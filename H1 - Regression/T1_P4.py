#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# Basis functions

def basis_a (xx):
    return np.vstack((np.ones(xx.shape), xx, xx ** 2, xx ** 3, xx ** 4, xx ** 5)).T

def basis_b (xx):
    return np.vstack((np.ones(xx.shape), np.exp(np.square(xx - 1960.0) / (-25.0)), 
                                         np.exp(np.square(xx - 1965.0) / (-25.0)),
                                         np.exp(np.square(xx - 1970.0) / (-25.0)),
                                         np.exp(np.square(xx - 1975.0) / (-25.0)),
                                         np.exp(np.square(xx - 1980.0) / (-25.0)),
                                         np.exp(np.square(xx - 1985.0) / (-25.0)),
                                         np.exp(np.square(xx - 1990.0) / (-25.0)),
                                         np.exp(np.square(xx - 1995.0) / (-25.0)),
                                         np.exp(np.square(xx - 2000.0) / (-25.0)),
                                         np.exp(np.square(xx - 2005.0) / (-25.0)),
                                         np.exp(np.square(xx - 2010.0) / (-25.0)))).T

def basis_c (xx):
    return np.vstack((np.ones(xx.shape), np.cos(xx / 1.0), np.cos(xx / 2.0),
                                         np.cos(xx / 3.0), np.cos(xx / 4.0),
                                         np.cos(xx / 5.0))).T

def basis_d (xx):
    return np.vstack((np.ones(xx.shape), np.cos(xx / 1.0), np.cos(xx / 2.0),
                                         np.cos(xx / 3.0), np.cos(xx / 4.0),
                                         np.cos(xx / 5.0), np.cos(xx / 6.0),
                                         np.cos(xx / 7.0), np.cos(xx / 8.0),
                                         np.cos(xx / 9.0), np.cos(xx / 10.0),
                                         np.cos(xx / 11.0), np.cos(xx / 12.0),
                                         np.cos(xx / 13.0), np.cos(xx / 14.0),
                                         np.cos(xx / 15.0), np.cos(xx / 16.0),
                                         np.cos(xx / 17.0), np.cos(xx / 18.0),
                                         np.cos(xx / 19.0), np.cos(xx / 20.0),
                                         np.cos(xx / 21.0), np.cos(xx / 22.0),
                                         np.cos(xx / 23.0), np.cos(xx / 24.0),
                                         np.cos(xx / 25.0))).T


# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
    
    if part == "a":
        return basis_a(xx)    

    if part == "b":
        return basis_b(xx)    

    if part == "c":
        return basis_c(xx)    
        
    if part == "d":
        return basis_d(xx)    

    return None

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, find_weights(X,Y))

# Loss function

def calc_loss(x, y):
    sum = 0.0
    for i in range(len(x)):
        sum += 0.5 * ((y[i] - np.dot(find_weights(x, y), x[i])) ** 2.0)
    return sum

# Plot and report sum of squared error for each basis 

for i in ['a', 'b', 'c', 'd']:
    X = make_basis(years, part = i)
    grid_X = make_basis(grid_years, part = i)
    Yhat = np.dot(grid_X, find_weights(X,Y))
    loss = calc_loss(X,Y)
    print("Loss Republicans by Year Basis " + i + ": " + str(loss))
    plt.plot(years, republican_counts, 'o', grid_years, Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Number of Republicans by Year: Basis " + i)
    plt.savefig('year' + i + '.png')
    plt.show()

grid_spots = np.linspace(0, 160, 160)
grid_spots_set = sunspot_counts[years<last_year]
grid_republican_set = republican_counts[years<last_year]

for i in ['a', 'c', 'd']:
    X = make_basis(grid_spots_set, part = i, is_years = False)
    grid_X = make_basis(grid_spots, part = i, is_years = False)
    Yhat = np.dot(grid_X, find_weights(X, grid_republican_set))
    loss = calc_loss(X, grid_republican_set)
    print("Loss Republicans by Sunspots Basis " + i + ": " + str(loss))
    plt.plot(grid_spots_set, grid_republican_set, 'o', grid_spots, Yhat, '-')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Number of Republicans by Sunspots: Basis " + i)
    plt.savefig('sunspot' + i + '.png')
    plt.show()