# TYRE RECOGNIZER (TR) - HEROKU APP
----

## Overview

This project aims to create an useful application which detects brand and size of any car tyre with machine learning algorithms. The application developed here is a prototype; the final version will be created for mobile devices like smartphones or tablets.

## Purpose

The goal of this app is to provide a useful service to users. The main goals are:

* B2C side: the application detects and reads the information from the tyre and will give to the user an estimate of the price of that kind of tyre. Alternatively, it will extract a price range or give the cheapest and the most expensive tyres with those features.

* B2B side: the application allows the mechanic or the tyre repairer to know if in the warehouse there are stocks of that tyre. In order to make this, the app should be connected to a database. If necessary, the user can note the absence of stocks, or make an order.

## Techniques

The code was written entirely with Python; in this project I used techniques, algorithms and functions that will be explained shortly below and more in detail in the Method section into the website. For simplicity, I considered in the project only 4 brands: Bridgestone, Continental, Michelin and Pirelli.

* At the beginning the application reads the image and looks for circles, in order to find the tyre's center. Then, it straightens the tyre into a rectangle. To make this, I used OpenCV functions;

* In order to establish the brand name, the straighted tyre is given as input to a neural network classifier;

* After that, I found the size with the help of the Google OCR;

* Finally, with scraping-techniques I got information from the site GommaDiretto.it about the cheapest and the most expensive tyres with the features found before.