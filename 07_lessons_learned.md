---
layout: page
title: Lessons Learned
nav_order: 7
---
  * Large data sets are a pain in the butt. Do not underestimate the size of your
    data set while simultaneously overestimating the performance of your hardware.
    And not even think about transferring it over the internet
  * START WITH SMALL DATASETS. I didn't. I regret it.
  * Always have HPC hardware at your disposal :)
  * Working with special hardware is only fun, if there are libraries to utilize it
  * Documentation and comments are very important, especially in parts with more complex math
  * maybe do the calculation below, first ...

# Cost of the project

Wrapping up, I just want to briefly look at the cost associated with that project.
Because besides the hours of work that gone into, this project is everything but
free.
  * _Labor cost to program, debug and fix things._
  * Parsing 13 million frames took about 50000 core-hours (runtime on each core
    added up). And when look at prices of cloud computing (Amazon AWS), you pay between 20 and 50
    cents per core-hour. Which are between US$ 10000 and US$ 25000 just for the first
    step. I'm happy I didn't had to pay that.
  * 13 million frame take up about 2TB of storage which are 46$/month in AWS pricing.
    I fortunately hadn't to pay that either
  * Displaying the result in the internet is: (and this I have to pay)
    * US$ 0.005/GB/Month storage of the result
    * US$ 0.01GB/GB/Month Download bandwith
    * € 6/Month for the secound caching server
    * € 1/Month for the domain (paid annually)
  * And as already mentioned if you'd wanted to print this you would pay € 6/m²
    to a total of € 35000 for a 1.2MP*1.2MP picture
