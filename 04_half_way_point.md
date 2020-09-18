---
layout: page
title: Halfway mark
nav_order: 4
---
# what we've learned so far?
At this point my `dominantColor.py` script is already running for 8 days and is
estimated to run another 8 days. Uff. I might have underestimated the amount of
data or underestimated the performance of the server.

Either way. I was curios how things look so far. So I grabbed the data already
generated to visualize how far I am.

And it looks like this:

![Halfway]({{site.baseurl}}/assets/images/halfway.png)

What does this picture show, you might ask? Well it shows the dominant Color of
each frame already processed, grouped by the video it belongs to. An interesting
observation is that the generation of such a distinct pattern, would mean that
each video seams to be more or less consistent in terms of color over its
full length. Makes sense from a filmmaking and design standpoint, but interesting
to see it visualized nonetheless.

Another thing that this visualization shows, whether the desired meta-image
is going to be generatable or not, by comparing the histograms, either by
generating it with matplotlib or by loading the image into to a graphics software.

So here we have the histogram of the picture generated with GIMP

![Source Histogram]({{site.baseurl}}/assets/images/hist_src.jpg)

With this first particular data set the goals was to generate this image

![Destination 1 Image]({{site.baseurl}}/assets/images/dest1_unus_annus.jpg)
_Credit: [Unus Annus](https://www.youtube.com/channel/UCIcgBZ9hEJxHv6r_jDYOMqg) -
[Ethan "CrankGameplays" Nestor](https://www.youtube.com/channel/UC2S7CGceq5tsuhcVyACdA3g)
and [Mark "Markiplier" Fischbach](https://www.youtube.com/channel/UC7_YxT-KID8kRbqZo7MyscQ)_

The image look distorted on purpose, because the source images are not square, therefore
distorting the meta-image back to its original aspect ratio of 16:9. But as you
could probably already guess and the histogram shows:

![Destination 1 Histogram]({{site.baseurl}}/assets/images/hist_dst1.jpg)

The data will not be optimal for this image. There might be a way around by
abstracting black and white to dark and light or using frames more than once.
But another image should probably be considered. E.g. this one looks a lot more promising

![Destination 2 Image]({{site.baseurl}}/assets/images/EgSMMI1XoAEt5aG.jpeg)
_Credit: [Unus Annus](https://www.youtube.com/channel/UCIcgBZ9hEJxHv6r_jDYOMqg) -
[Ethan "CrankGameplays" Nestor](https://www.youtube.com/channel/UC2S7CGceq5tsuhcVyACdA3g)
and [Mark "Markiplier" Fischbach](https://www.youtube.com/channel/UC7_YxT-KID8kRbqZo7MyscQ)_

![Destination 2 Histogram]({{site.baseurl}}/assets/images/hist_dst2.jpg)

Creating the graphic to check was quite easy. The script iterates overall data
set, decoding the packed integer, drawing the corresponding pixel in the destination
image. The and aspect ratio is calculated manually, by just taking the array length
and factorize it. This is not optimal, because not portable, but it works for a
quick look.

```python
#!/usr/bin/env python3
import sys, os

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import cv2


dirn = os.path.dirname(os.path.realpath(__file__))

array = []

for f in os.listdir(dirn):
    if f.endswith(".sqlite"):
        try:
            print("{}\n".format(f))
            connection = sqlite3.connect(f)
            c = connection.cursor()
            for row in c.execute("SELECT color1 FROM images"):
                r = row[0] >> 16 & 255
                g = row[0] >> 8  & 255
                b = row[0]       & 255
                array.append([r, g, b])
            connection.close()
        except:
            pass

print(len(array))

array = np.array(array[:9819452])
print(array.shape)
array = array.reshape((6689, 1468, 3)).astype("int8")
print(array.shape)
#print(array)


#img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

cv2.imshow("", array)
cv2.waitKey(0)
cv2.destroyAllWindows()

clr = ('b','g','r')
for i,col in enumerate(clr):
    histr = cv2.calcHist([array],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.set_xlim(0,256)

plt.show()
```
