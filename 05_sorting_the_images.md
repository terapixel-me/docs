---
layout: page
title: Actually tileing the images
teaser: After two weeks of quantization, we are finally ready to assign images to the individual pixels.
nav_order: 5
toc: true
---
# Creating the Image-tiles

To begin with I decided to try these three images

![Images to tile]({{site.baseurl}}/assets/images/tiled_tries.png)

Judging by my color chart again, none of them is a very good match. But because
of a lack of better alternatives I'll try it anyway.

![Figure 1]({{site.baseurl}}/assets/images/Figure_1.png)

(_13 MP is a lot, that's why the color chart is scaled. And I'm not sure why the
GIMP histogram does not match the matplotlib one_)

## Developing the algorithm

What does it mean for a picture to be considered for one of the pixels? It means
that, if reduced to its dominant color, that the color is either a perfect match
or the color is at least very close to the desired one. And as an added bonus, it
would be nice to avoid using frames multiple times as much as possible.

So for each pixel of the meta-image we need to find either a perfect match or a
close alternative. Then save which picture we used to not use it again. In
pseudo-code, it would look something like this:

```
COLORS = DICTIONARY(color, LIST(filename))
FILES_USED = LIST(filename)

FILE_NAMES = MATRIX()

for PIXEL in META_PICTURE do
    if PIXEL in COLORS:
        if (COLORS[PIXEL.COLOR] \ FILES_USED) = ∅:
            do
                ALTERNATIVE = FIND_CLOSE_COLOR(PIXEL.COLOR)
            while (COLORS[ALTERNATIVE] \ FILES_USED) = ∅
            FILE = (COLORS[ALTERNATIVE] \ FILES_USED)[RANDOM_INDEX]
        else
            FILE = (COLORS[PIXEL.COLOR] \ FILES_USED)[RANDOM_INDEX]
    else:
        do
            ALTERNATIVE = FIND_CLOSE_COLOR(PIXEL.COLOR)
        while (COLORS[ALTERNATIVE] \ FILES_USED) = ∅
        FILE = (COLORS[ALTERNATIVE] \ FILES_USED)[RANDOM_INDEX]
    end
    FILES_USED = FILES_USED ∪ {FILE}
    FILE_NAMES[PIXEL.COORDIANTES] = FILE
next
```

Actually programming to in this way, would be extremely inefficient and slow,
because for each iteration multiple lists and set would need to be traversed and
compared. And by now we don't even know how `FIND_CLOSE_COLOR()` would work.

### What are close colors

Two colors would be considered close if one of them is slightly lighter or darker
than the other or even if the Hue is slightly of. E.g.

![Close Colors]({{site.baseurl}}/assets/images/close_colors_2.png)

So obviously the reds are quite close and would not make that big of a difference
when used interchangeably. Whereas the blue and the green are not close at all.
But how to calculate the differences?

When representing the colors as vectors in RGB color space
{% raw %}
$
\overrightarrow{{red}\_1} = \begin{bmatrix} r_1 \\\\ g_1 \\\\ b_1 \end{bmatrix} = \begin{bmatrix} 229.9 \\\\ 29.4 \\\\ 29.4 \end{bmatrix},
\overrightarrow{{red}\_2} = \begin{bmatrix} r_2 \\\\ g_2 \\\\ b_2 \end{bmatrix} = \begin{bmatrix} 229.9 \\\\ 29.4 \\\\ 63.1 \end{bmatrix},
$

$
\overrightarrow{{red}\_3} = \begin{bmatrix} r_3 \\\\ g_3 \\\\ b_3 \end{bmatrix} = \begin{bmatrix} 255.0 \\\\ 32.6 \\\\ 32.6 \end{bmatrix},
\overrightarrow{{red}\_4} = \begin{bmatrix} r_4 \\\\ g_4 \\\\ b_4 \end{bmatrix} = \begin{bmatrix} 229.9 \\\\ 78.6 \\\\ 78.6 \end{bmatrix}
$

$
\overrightarrow{blue} = \begin{bmatrix} r_5 \\\\ g_5 \\\\ b_5 \end{bmatrix} = \begin{bmatrix} 29.4 \\\\ 34.2 \\\\ 229.9 \end{bmatrix},
\overrightarrow{green} = \begin{bmatrix} r_6 \\\\ g_6 \\\\ b_6 \end{bmatrix} = \begin{bmatrix} 29.4 \\\\ 229.9 \\\\ 64.8 \end{bmatrix}
$
{% endraw %}

than _a_ distance could be the Euclidean distance $ d(\vec{c_1},\vec{c_2}) = d(\vec{c_2},\vec{c_1}) $:

{% raw %}
$ d(\vec{c_1},\vec{c_2}) = \sqrt{\sum_{i=1}^n ({c_1}_i-{c_2}_i)^2} = \sqrt{ (r\_{c_1}-r\_{c_2})^2 + (g\_{c_1}-g\_{c_2})^2 + (b\_{c_1}-b\_{c_2})^2 } $
{% endraw %}

{% include note.html content="Euclidean distance, in mathematics, is only one of several
different ways of measuring distances between vectors. For the purpose of this documentation
it is just the easiest one to imagine, as it describes the length of a straight line
between the vectors. The code later will not use this metric. The minkowski-metric will be used.
\\
Look here https://en.wikipedia.org/wiki/Metric_(mathematics) to learn more about metrics" %}

And the distances to $ \overrightarrow{\{red\}\_1} $ are:
{% raw %}
$ d(\overrightarrow{{red}\_1},\overrightarrow{{red}\_1}) = 0.0 $  

$ d(\overrightarrow{{red}\_1},\overrightarrow{{red}\_2}) = 33.7 $  

$ d(\overrightarrow{{red}\_1},\overrightarrow{{red}\_3}) \approx 25.5 $  

$ d(\overrightarrow{{red}\_1},\overrightarrow{{red}\_4}) \approx 69.6 $  (_nice!_)

$ d(\overrightarrow{{red}\_1},\overrightarrow{blue}) \approx 283,6 $  

$ d(\overrightarrow{{red}\_1},\overrightarrow{green}) \approx 285.8 $  
{% endraw %}

As expected the distance to green and blue is way higher than to the other shades
of red. The problem is this distances can only be measured pairwise. This would
mean, if we want to find an alternative for a particular color, we would need to
measure the distance to each of the available colors. In order to select, the
closest unused one. And doing this within the code-example from above would make
it even more inefficient and it would take ages to get the pictures for all pixels.

### K-Nearest Neighbors

The problem described above, is also known as "K-Nearest Neighbors" in computer science.

> In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method proposed by Thomas Cover used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.
>
> In k-NN classification, the output is a class membership. __An object is classified by a plurality vote of its neighbors__, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
>
> In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
>
> k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation. Since this algorithm relies on distance for classification, normalizing the training data can improve its accuracy dramatically.

Wikipedia: [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

Which means, out of the training samples in the feature space, a new sample is
classified by its k closed neighbors. Closeness is measured with a distance-metric
(e.g. Euclidean distance).

This helps us when using the available colors as our training samples and the color,
we want to find an alternative for, as the sample classified by its neighbors.

Again as this is a very common thing to do, there are plenty of libraries available.
I'm going to use IBMs SnapML Toolkit again.

But so far, so good. We have a library, and we know which algorithm to use. But how
to use it? The easiest and most straight forward idea would to use all 13 million
images colors as feature space and for each color of the meta-image search the
k neighbors closed to it. k is the maximum number a color appears in the
meta-image.

```python
for f in os.listdir(os.path.join(dirn,"tp_data")):
    if f.endswith(".sqlite"):
        #print("{}\n".format(os.path.join(dirn,"tp_data",f)))
        connection = sqlite3.connect(os.path.join(dirn,"tp_data",f))
        c = connection.cursor()
        for row in c.execute("SELECT color1 FROM images"):
            r = row[0] >> 16 & 255
            g = row[0] >> 8  & 255
            b = row[0]       & 255
            array.append([r, g, b])
        connection.close()


array = np.array(array)

img = cv2.cvtColor(cv2.imread("0179.jpg"),cv2.COLOR_BGR2RGB)

old_shape = img.shape

img = img.reshape((old_shape[0] * old_shape[1], old_shape[2]))
print(img)

print("np.unique(img)")
search, inverse, counts = np.unique(img, return_counts=True, return_inverse=True, axis=0)

print("{} pixels, {} candidates".format(len(img), len(array)))
print("search =")
print(search)
print("counts =")
print(counts)
print("inverse =")
print(inverse)

print("fit(array) clr/{}".format(counts.max()))
nn_clr = NearestNeighbors(n_neighbors=counts.max(), algorithm='auto', n_jobs=-1).fit(array)
print(nn_clr)
print("kneighbors(search)")
distances, indices = nn_clr.kneighbors(search)

print("indices = ")
print(indices)
```

But this is going to search the ~100000 nearest of ~200000 samples in a 13 million
sample search space. This consumes more memory than you could ever have in your
computer.

So we need to find a more sophisticated way to search for nearest neighbors.

As already mentioned colors could also match perfectly, eliminating the need to
search for near neighbors. So a color can have:

- a sufficient number of perfect matches
- an insufficient number of perfect matches, requiring alternatives for some pixels
- no matches at all, requiring alternatives for all pixels of this color

With that in mind we can shrink the search space and amount of samples needed to
be classified.

{% raw %}
$ I = \\{\text{distinct set of colors in the meta-image}\\} $

$ I' = I \times \mathbb{N} = \\{(\text{color}, \text{amount in image})\\} $

$ A = \\{\text{distinct set of available colors}\\}$

$ A' = A \times \mathbb{N} = \\{(\text{color}, \text{amount available})\\} $

$ I \setminus A = \text{required colors without matches} $

$ A \setminus I = \text{colors that are not required and available as alternatives} $

$ I \cap A = \text{color with matches}$

$ R_A = \\{ (c, a - b) \| (c, a) \in I', (c, b) \in A', a < b \\} = \text{available colors, not required that often} $

$ R_I = \\{ (c, b - a) \| (c, a) \in I', (c, b) \in A', a > b \\} = \text{colors required more than available}$

$ S = (A \setminus I) \cup \\{ c \| (c,a) \in R_A \\} = \text{search space} $

$ F = (I \setminus A) \cup \\{ c \| (c,a) \in R_I \\} = \text{color to search} $

$ k = max[max_a(\\{ a \| (c,a) \in R_I \\}) / avg_a(\\{ a \| (c,a) \in R_A \\}), max_a(\\{ a \| (c,a) \in I' \land c \notin A \\}) / avg_a(\\{ a \| (c,a) \in A' \land c \notin I  \\})]$
{% endraw %}

With this set-arithmetic out search $ \|S\| \ll \|\{\mathbb{N}^3\}^{13\times{}10^6}\| $ is
now much smaller than _all_ available pixels. Also, the number of points to find $ \|F\| \ll \|\{\mathbb{N}^3\}^{2\times{}10^6}\| $.

The $ k $ I choose is a bit a wired one.

$max_a(\\{ a \| (c,a) \in R_I \\})$ is the
maximum of missing perfect matches. $avg_a(\\{ a \| (c,a) \in R_A \\})$ is the average
amount for available colors. The ratio between the two, is the average amount of colors
I need to consider to have enough pixels to find an alternative for the unmatched ones.

$max_a(\\{ a \| (c,a) \in I' \land c \notin A \\})$ is the maximum color amount
of the unmatched colors. $avg_a(\\{ a \| (c,a) \in A' \land c \notin I  \\})$ is
amount of the available colors, that are unmatched. The ratio between the two, is the
average amount of colors I need to consider finding an alternative for the colors that
have no perfect matches.

And the maximum of these two is the number of colors to consider, to have
on average a sufficient amount of alternatives available.

This all is done because individual colors are available and required multiple times.
And with this heuristic we avoid needing to add duplicates in the search space or
the list of sampled points. Unfortunately this heuristic is not perfect and it can
happen that not enough alternatives are available, requiring duplicate frames. But
should only happen for at most 50% of the unmatched pixels.

The concrete numbers for the first picture were

$ \|S\| = \{\mathbb{N}^3\}^\{1083388\} $, $ \|F\| = \{\mathbb{N}^3\}^{49465} $ and $ k = 281 $

49465 out of 76826 colors don't have perfect matches. For at least 50% of which
we should have unique machetes. Which is not great but good enough.

# Pythonize the theory

Now that we are done with the theory, we just need to put all of this into code

Let's begin with the imports

```python
import sys, os
import sqlite3
import numpy as np
import cv2
import pandas as pd
import cudf
# from sklearn.neighbors import NearestNeighbors
from pai4sk.neighbors import NearestNeighbors
from collections import defaultdict
import random
import functools
import json
import h5py
from datetime import datetime
import copy
```

Besides `h5py` and `copy`, nothing special here. Basic stuff and math and image
processing stuff. We will use HDF5 to save our calculation for later or to recover
from crashes. `copy` is required to create working sets that can be safely modified
during runtime, you'll see later.

Next on some "configuration" and helper Functions

```python
print = functools.partial(print, flush=True, file=sys.stderr)

# dirn = os.path.dirname(os.path.realpath(__file__))
dirn = "/path/to/tp_data"

img_to_process = sys.argv[1]

print(img_to_process)

# color array
array = []
# filenames providing a color
col_dict = defaultdict(list)


def arrpack(arr):
    return (arr[0] & 255) << 16 | (arr[1] & 255) << 8 | (arr[2] & 255)


def arrunpack(num):
    return [(num >> 16) & 255, (num >> 8) & 255, (num & 255)]


def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d' % i: df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf
```

`arrpack()` and `arrunpack()` are to concatenate and split apart the RGB vectors.
This is very handy because 1 dimension is sometimes much more easy to work with than 3.

Now we read the sqlite-files created previously containing the filenames and colors.

```python
for f in os.listdir(os.path.join(dirn, "tp_data")):
    if f.endswith(".sqlite"):
        print(".", end='', flush=True)
        try:
            connection = sqlite3.connect(os.path.join(dirn, "tp_data", f))
            c = connection.cursor()
            for row in c.execute("SELECT color1, imgname FROM images"):
                array.append(row[0])
                # populate the color->filenames dict
                col_dict[row[0]].append(row[1])
            connection.close()
        except sqlite3.OperationalError:
            pass
```

The next part is just to read data from any previous run, if existent.

```python
print("")
if os.path.exists("{}-dump.hdf5".format(img_to_process)):
    print("dump.hdf5 exists, reading data...")
    with h5py.File("{}-dump.hdf5".format(img_to_process), 'r') as f:
        print(list(f.keys()))
        array = np.array(f["array"])
        array_p = np.array(f["array_p"])
        img = np.array(f["img"])
        img_p = np.array(f["img_p"])
        img_p_u = np.array(f["img_p_u"])
        img_p_u_inverse = np.array(f["img_p_u_inverse"])
        img_p_u_counts = np.array(f["img_p_u_counts"])
        array_p_u = np.array(f["array_p_u"])
        array_p_u_inverse = np.array(f["array_p_u_inverse"])
        array_p_u_counts = np.array(f["array_p_u_counts"])
        img_p_u_um = np.array(f["img_p_u_um"])
        img_p_u_um_counts = np.array(f["img_p_u_um_counts"])
        array_p_u_um = np.array(f["array_p_u_um"])
        array_p_u_um_counts = np.array(f["array_p_u_um_counts"])
        data_p_u_m = np.array(f["data_p_u_m"])
        img_p_u_mi = np.array(f["img_p_u_mi"])
        array_p_u_mi = np.array(f["array_p_u_mi"])
        img_rest = np.array(f["img_rest"])
        img_rest_counts = np.array(f["img_rest_counts"])
        array_rest = np.array(f["array_rest"])
        array_rest_counts = np.array(f["array_rest_counts"])
        search_space_ = np.array(f["search_space_"])
        search_space = np.array(f["search_space"])
        find_points_ = np.array(f["find_points_"])
        find_points = np.array(f["find_points"])
        distances = np.array(f["distances"])
        indices = np.array(f["indices"])
        old_shape = cv2.imread(img_to_process).shape
else:
    ...
```

If there is no previous data we need to calculate it first. This is the actual math
part from above.

```python
  # this is all part of the else block
  array = np.array(array)

  print("cv2.imread()")
  img = cv2.cvtColor(cv2.imread(img_to_process), cv2.COLOR_BGR2RGB)

  old_shape = img.shape
  print("img.reshape()")
  img = img.reshape((old_shape[0] * old_shape[1], old_shape[2]))

  print("arrpack()")
  # array_p = np.array([arrpack(p) for p in array])
  array_p = array # left over for compatibility
  img_p = np.array([arrpack(p) for p in img])

  # creating the distinct sets
  print("np.unique(img)")
  img_p_u, img_p_u_inverse, img_p_u_counts = np.unique(img_p, return_counts=True, return_inverse=True, axis=0)
  print("np.unique(array)")
  array_p_u, array_p_u_inverse, array_p_u_counts = np.unique(array_p, return_counts=True, return_inverse=True, axis=0)

  # I \ A
  print("np.setdiff1d(img_p_u, array_p_u)")
  img_p_u_um = np.setdiff1d(img_p_u, array_p_u, assume_unique=True)
  img_p_u_um_counts = img_p_u_counts[np.in1d(img_p_u, img_p_u_um)]

  # A \ I
  print("np.setdiff1d(array_p_u, img_p_u)")
  array_p_u_um = np.setdiff1d(array_p_u, img_p_u, assume_unique=True)
  array_p_u_um_counts = array_p_u_counts[np.in1d(array_p_u, array_p_u_um)]

  # I ∩ A
  print("np.intersect1d(img_p_u, array_p_u)")
  data_p_u_m, img_p_u_mi, array_p_u_mi = np.intersect1d(img_p_u, array_p_u, assume_unique=True, return_indices=True)

  img_rest = []
  img_rest_counts = []
  array_rest = []
  array_rest_counts = []

  print("calulation of rests")
  for (dp, i, j) in zip(data_p_u_m, img_p_u_mi, array_p_u_mi):
      if img_p_u_counts[i] < array_p_u_counts[j]:
          array_rest.append(dp)
          array_rest_counts.append(array_p_u_counts[j] - img_p_u_counts[i])
      elif img_p_u_counts[i] > array_p_u_counts[j]:
          img_rest.append(dp)
          img_rest_counts.append(img_p_u_counts[i] - array_p_u_counts[j])

  img_rest = np.array(img_rest)
  img_rest_counts = np.array(img_rest_counts)
  array_rest = np.array(array_rest)
  array_rest_counts = np.array(array_rest_counts)

  print("{} pixels, {} candidates".format(img.shape[0], array.shape[0]))
  print("{} colors required, {} colors available".format(img_p_u.shape[0], array_p_u.shape[0]))
  print("{} colors matched, {} matched low, {} matched high".format(data_p_u_m.shape[0], img_rest.shape[0],
                                                                    array_rest.shape[0]))
  print("{} colors unmatched in img, {} colors unmatched in array".format(img_p_u_um.shape[0], array_p_u_um.shape[0]))

  search_space_ = np.union1d(array_p_u_um, array_rest)
  search_space = np.array([arrunpack(p) for p in search_space_])

  find_points_ = np.union1d(img_p_u_um, img_rest)
  find_points = np.array([arrunpack(p) for p in find_points_])

  # the "k"
  ntf = int(max(img_rest_counts.max() / np.average(array_rest_counts),
                img_p_u_um_counts.max() / np.average(array_p_u_um_counts)))

  print("fit(search_space) clr/{}".format(ntf))
  nn = NearestNeighbors(n_neighbors=ntf, algorithm='auto', n_jobs=155).fit(search_space)
  print(nn)
  print("kneighbors(find_points)")
  distances, indices = nn.kneighbors(find_points)

  print(distances)
  print(indices)

```

Because depending on the image, `NearestNeighbors` can take one or two eternities we now
save what we have calculated by now

```python
  # still in the else block
  print("saving data to {}-dump.hdf5".format(img_to_process))
      with h5py.File("{}-dump.hdf5".format(img_to_process), "w") as f:
          f.create_dataset("array", data=array)
          f.create_dataset("array_p", data=array_p)
          f.create_dataset("img", data=img)
          f.create_dataset("img_p", data=img_p)
          f.create_dataset("img_p_u", data=img_p_u)
          f.create_dataset("img_p_u_inverse", data=img_p_u_inverse)
          f.create_dataset("img_p_u_counts", data=img_p_u_counts)
          f.create_dataset("array_p_u", data=array_p_u)
          f.create_dataset("array_p_u_inverse", data=array_p_u_inverse)
          f.create_dataset("array_p_u_counts", data=array_p_u_counts)
          f.create_dataset("img_p_u_um", data=img_p_u_um)
          f.create_dataset("img_p_u_um_counts", data=img_p_u_um_counts)
          f.create_dataset("array_p_u_um", data=array_p_u_um)
          f.create_dataset("array_p_u_um_counts", data=array_p_u_um_counts)
          f.create_dataset("data_p_u_m", data=data_p_u_m)
          f.create_dataset("img_p_u_mi", data=img_p_u_mi)
          f.create_dataset("array_p_u_mi", data=array_p_u_mi)
          f.create_dataset("img_rest", data=img_rest)
          f.create_dataset("img_rest_counts", data=img_rest_counts)
          f.create_dataset("array_rest", data=array_rest)
          f.create_dataset("array_rest_counts", data=array_rest_counts)
          f.create_dataset("search_space_", data=search_space_)
          f.create_dataset("search_space", data=search_space)
          f.create_dataset("find_points_", data=find_points_)
          f.create_dataset("find_points", data=find_points)
          f.create_dataset("distances", data=distances)
          f.create_dataset("indices", data=indices)
```

Now for the fun part. Selecting the actual file for a pixel. The basic assumption
here is, that there are now 3 cases for each pixel
- case 1: there is a sufficient number of perfect matches -> they will all be reserved for this case
- case 2: there are perfect matches, but not enough -> we need an alternate
- case 3: there are only approximations available
- case 2a and 3a: the number of approximations is too low -> we need duplicates.

We save the file-name to an array and desperately (because numpy is wired and h5py doesn't like unicode) try saving this array to get the
pictures ready for upload.

```python
new_img = []
new_img_fns = []

fns_blacklist = []

col_times_used = defaultdict(int)
alt_col_times_used = defaultdict(int)
# array_p_u_counts

duplicats_used = False

col_dict_work = copy.deepcopy(col_dict)

print("start matching phase")
try:
    for (pix, ctr) in zip(img_p_u_inverse, range(img_p_u_inverse.shape[0])):
        start = datetime.now()
        # pix = index of color in img_p_u
        # case 1: there is a sufficient number of perfect matches
        # case 2: there are perfect matches, but not enough
        # case 3: there are only approximations available
        # case 2a and 3a: the number of approximations is too low -> we are fucked!
        print("Pixel {} ".format(ctr), end='', flush=True)
        curr_col = img_p_u[pix]
        print("Color {} ".format(curr_col), end='', flush=True)
        curr_col_in_intersect = (data_p_u_m == curr_col).nonzero()[0]
        if curr_col_in_intersect.shape[0] > 0:
            print("matched ", end='', flush=True)
            # our color is in the intersection of img_p_u and array_p_u
            ind_in_intersect = curr_col_in_intersect[0]
            colamount_in_img = img_p_u_counts[pix]
            colamount_in_arr = array_p_u_counts[array_p_u_mi[ind_in_intersect]]
            if colamount_in_img <= colamount_in_arr:
                # we are in case 1
                col_times_used[curr_col] += 1
                print("perfectly ", end='', flush=True)
                new_img.append(arrunpack(curr_col))
                # choose_from = list(set(col_dict[curr_col]) - set(fns_blacklist))
                choose_from = col_dict_work[curr_col]
                print("{} avail for color, {} used for color, {} left for color, {} total blacklisted ".format(
                    len(col_dict[curr_col]), col_times_used[curr_col], len(choose_from), len(fns_blacklist)), end='',
                      flush=True)
                fn = choose_from.pop(random.randrange(len(choose_from)))
                col_dict_work[curr_col] = choose_from
                print("file {} ".format(fn), end='', flush=True)
                fns_blacklist.append(fn)
                new_img_fns.append(fn)
            else:
                # we are in case 2
                num_of_perf_match = colamount_in_arr
                print("low ", end='', flush=True)
                # can we still use a perfect one?
                if col_times_used[curr_col] < num_of_perf_match:
                    # flip a slightly biased (2:3) coin to decide whether to use one
                    print("perfects available ", end='', flush=True)
                    if random.randint(0, 2) % 2 == 0:
                        print("using perfect ", end='', flush=True)
                        col_times_used[curr_col] += 1
                        new_img.append(arrunpack(curr_col))
                        # choose_from = list(set(col_dict[curr_col]) - set(fns_blacklist))
                        choose_from = col_dict_work[curr_col]
                        print("{} used for color, {} left for color, {} total blacklisted ".format(
                            col_times_used[curr_col], len(choose_from), len(fns_blacklist)), end='', flush=True)
                        fn = choose_from.pop(random.randrange(len(choose_from)))
                        col_dict_work[curr_col] = choose_from
                        print("file {} ".format(fn), end='', flush=True)
                        fns_blacklist.append(fn)
                        new_img_fns.append(fn)
                    else:
                        # use an alternative
                        print("using alt ", end='', flush=True)
                        ind_in_find_points = (find_points_ == curr_col).nonzero()[0][0]
                        alt_indices = indices[ind_in_find_points]  # list of indexes in search space
                        alt_colors = search_space_[np.sort(alt_indices)]
                        col_counts = array_p_u_counts[np.isin(array_p_u, alt_colors)]
                        col_assigned = False
                        for (alt_col, alt_col_count) in zip(alt_colors, col_counts):
                            is_reserved = False
                            if (img_p_u == alt_col).nonzero()[0].shape[0] > 0:
                                # alternate color candiate is also in img
                                h = img_p_u_counts[(img_p_u == alt_col).nonzero()[0][0]]
                                if h < alt_col_count - alt_col_times_used[alt_col]:
                                    is_reserved = False
                                else:
                                    # we still need this color for remaining perfect matches
                                    is_reserved = True
                            if alt_col_times_used[alt_col] < alt_col_count and not is_reserved:
                                print("{} ".format(alt_col), end='', flush=True)
                                alt_col_times_used[alt_col] += 1
                                new_img.append(arrunpack(alt_col))
                                # choose_from = list(set(col_dict[alt_col]) - set(fns_blacklist))
                                choose_from = col_dict_work[alt_col]
                                print("{} avail for alt, {} used of alt, {} left for alt, {} total blacklisted ".format(
                                    alt_col_count, alt_col_times_used[alt_col], len(choose_from), len(fns_blacklist)),
                                      end='', flush=True)
                                fn = choose_from.pop(random.randrange(len(choose_from)))
                                col_dict_work[alt_col] = choose_from
                                print("file {} ".format(fn), end='', flush=True)
                                fns_blacklist.append(fn)
                                new_img_fns.append(fn)
                                col_assigned = True
                                break
                        if not col_assigned:
                            print("failed ", end='', flush=True)
                            # we ran out of alternatives (case 2a)
                            if col_times_used[curr_col] < num_of_perf_match:
                                # but we have perfect matches left. TODO duplicate code
                                print("using perfect ", end='', flush=True)
                                col_times_used[curr_col] += 1
                                new_img.append(arrunpack(curr_col))
                                # choose_from = list(set(col_dict[curr_col]) - set(fns_blacklist))
                                choose_from = col_dict_work[curr_col]
                                print("{} used for color, {} left for color, {} total blacklisted ".format(
                                    col_times_used[curr_col], len(choose_from), len(fns_blacklist)), end='', flush=True)
                                fn = choose_from.pop(random.randrange(len(choose_from)))
                                col_dict_work[curr_col] = choose_from
                                print("file {} ".format(fn), end='', flush=True)
                                fns_blacklist.append(fn)
                                new_img_fns.append(fn)
                            else:
                                # we need to take a duplicate :(
                                print("using duplicate perfect ", end='', flush=True)
                                duplicate_used = True
                                new_img.append(arrunpack(curr_col))
                                fn = random.choice(col_dict[curr_col])
                                print("file {} ".format(fn), end='', flush=True)
                                new_img_fns.append(fn)
                else:
                    # no unused perfects available anymore. TODO duplicate code :(
                    # use an alternative
                    print("using alt ", end='', flush=True)
                    ind_in_find_points = (find_points_ == curr_col).nonzero()[0][0]
                    alt_indices = indices[ind_in_find_points]  # list of indexes in search space
                    alt_colors = search_space_[np.sort(alt_indices)]
                    col_counts = array_p_u_counts[np.isin(array_p_u, alt_colors)]
                    col_assigned = False
                    for (alt_col, alt_col_count) in zip(alt_colors, col_counts):
                        if (img_p_u == alt_col).nonzero()[0].shape[0] > 0:
                            is_reserved = False
                            # alternate color candiate is also in img
                            h = img_p_u_counts[(img_p_u == alt_col).nonzero()[0][0]]
                            if h < alt_col_count - alt_col_times_used[alt_col]:
                                is_reserved = False
                            else:
                                # we still need this color for remaining perfect matches
                                is_reserved = True
                        if alt_col_times_used[alt_col] < alt_col_count and not is_reserved:
                            print("{} ".format(alt_col), end='', flush=True)
                            alt_col_times_used[alt_col] += 1
                            new_img.append(arrunpack(alt_col))
                            # choose_from = list(set(col_dict[alt_col]) - set(fns_blacklist))
                            choose_from = col_dict_work[alt_col]
                            print("{} used of alt, {} left for alt, {} total blacklisted ".format(
                                alt_col_times_used[alt_col], len(choose_from), len(fns_blacklist)), end='', flush=True)
                            fn = choose_from.pop(random.randrange(len(choose_from)))
                            col_dict_work[alt_col] = choose_from
                            print("file {} ".format(fn), end='', flush=True)
                            fns_blacklist.append(fn)
                            new_img_fns.append(fn)
                            col_assigned = True
                            break
                    if not col_assigned:
                        print("failed ", end='', flush=True)
                        # we need to take a duplicate :(
                        print("using duplicate perfect ", end='', flush=True)
                        duplicate_used = True
                        new_img.append(arrunpack(curr_col))
                        fn = random.choice(col_dict[curr_col])
                        print("file {} ".format(fn), end='', flush=True)
                        new_img_fns.append(fn)

        else:
            # our color is in the intersection of img_p_u and array_p_u, therefore we are in case 3
            # TODO kinda dup of 2
            print("unmatched ", end='', flush=True)
            ind_in_find_points = (find_points_ == curr_col).nonzero()[0][0]
            alt_indices = indices[ind_in_find_points]  # list of indexes in search space
            alt_colors = search_space_[np.sort(alt_indices)]
            col_counts = array_p_u_counts[np.isin(array_p_u, alt_colors)]
            col_assigned = False
            for (alt_col, alt_col_count) in zip(alt_colors, col_counts):
                is_reserved = False
                if (img_p_u == alt_col).nonzero()[0].shape[0] > 0:
                    # alternate color candiate is also in img
                    h = img_p_u_counts[(img_p_u == alt_col).nonzero()[0][0]]
                    if h < alt_col_count - alt_col_times_used[alt_col]:
                        is_reserved = False
                    else:
                        # we still need this color for remaining perfect matches
                        is_reserved = True
                if alt_col_times_used[alt_col] < alt_col_count and not is_reserved:
                    alt_col_times_used[alt_col] += 1
                    new_img.append(arrunpack(alt_col))
                    # choose_from = list(set(col_dict[alt_col]) - set(fns_blacklist))
                    choose_from = col_dict_work[alt_col]
                    print("{} used of alt, {} left for alt, {} total blacklisted ".format(alt_col_times_used[alt_col],
                                                                                          len(choose_from),
                                                                                          len(fns_blacklist)), end='',
                          flush=True)
                    fn = choose_from.pop(random.randrange(len(choose_from)))
                    col_dict_work[alt_col] = choose_from
                    print("file {} ".format(fn), end='', flush=True)
                    fns_blacklist.append(fn)
                    new_img_fns.append(fn)
                    col_assigned = True
                    break
            if not col_assigned:
                # we ran out of alternatives (case 3a)
                # we need to take a duplicate :(
                print("duplicate required, ", end='', flush=True)
                duplicate_used = True
                col_to_use = random.choice(alt_colors)
                new_img.append(arrunpack(col_to_use))
                fn = random.choice(col_dict[col_to_use])
                print("file {} ".format(fn), end='', flush=True)
                new_img_fns.append(fn)
        print("")
        print(datetime.now() - start)
except:
    for _ in range(img.shape[0] - len(new_img)):
        new_img.append([0, 0, 0])
    new_img = np.array(new_img).astype("uint8")
    new_img = new_img.reshape(old_shape)
    new_img = cv2.cvtColor(new_img.astype("uint8"), cv2.COLOR_BGR2RGB)
    cv2.imwrite("{}-crash.jpg".format(img_to_process), new_img)
    print("delete {}-dump.hdf5 to start over".format(img_to_process))
    raise

print("\n\n")

if duplicate_used:
    print("BAD NEWS: duplicates needed to be used :(")

new_img = np.array(new_img)
new_img = new_img.reshape(old_shape)

new_img_fns = np.array(new_img_fns)
new_img_fns = new_img_fns.reshape((old_shape[0], old_shape[1]))

print(new_img_fns)
try:
    new_img = new_img.astype("uint8")
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("{}-result.jpg".format(img_to_process), new_img)
except:
    pass

try:
    np.savetxt("{}-dump.csv".format(img_to_process), new_img_fns, delimiter=',')
except:
    pass

try:
    np.save("{}-dump.npy".format(img_to_process), new_img_fns)
except:
    pass

try:
    new_img_fns.tofile("{}-dump.dat".format(img_to_process), sep=',')
except:
    pass

with h5py.File("{}-dump.hdf5".format(img_to_process), 'a') as f:
    try:
        f.create_dataset("new_img", data=new_img)
    except:
        pass
    new_img_fns_h5 = np.char.decode(new_img_fns.astype(np.bytes_), 'UTF-8')
    f.create_dataset("new_img_fns", data=new_img_fns_h5)

```

# Conclusion

The code is not prefect, there are many duplicate parts but there is not rally a
better alternative. Because its kinda duplicate, but not duplicate enough to
create a function.

Calculating the first picture, shown at the beginning took about 24h. And only

```python
np.save("{}-dump.npy".format(img_to_process), new_img_fns)
```

yielded a usable result, while trying to save the stuff...


<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>

<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)'], ['\`$', '$\`']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
