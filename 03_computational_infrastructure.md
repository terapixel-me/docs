---
layout: page
title: Computing stuff
nav_order: 3
toc: true
---
# Exploring the limits of your hardware

I already mentioned it in the previous chapter, with this project you hit the computational limits
of your computer pretty fast. And not only for generating the source data, but especially for
running the main part of the calculation.

Running the KMeans algorithm with a single frame takes between 1 and 3 seconds. Continuing the
napkin math from earlier, for about 11137500 frames this would take up to 33412500 seconds, about
386 days on a single CPU core. I don't have this much time, especially not with the
YouTube channel the fan-art is for.

# HPC to the Rescue

So when writing the next part, we need to do it in a way that big and fast computers can do the job.
And what make computers fast? Many CPU cores and plenty of RAM.

_Sidenote: in the repo you will find a file `dominantColor_set_slow.py` this file is reminiscent of the first
tries to run this on box standard computers. Therefor no tar archives are used, because my computer does not
care about the number of Inodes. But admins of large network storage usually do. I will not go deeper into
the specifics of that script, because it is just to slow, but it is very similar to what you will see in the
remaining of the chapter_

The program has 3 main parts
```python
def process(filen):
  ...

def processTar(tarball):
  ...

def main(folder, img, db, dst):
  ...

if __name__ == '__main__':
    print(sys.path)
    assert len(sys.argv) > 4
    folder = sys.argv[1]
    img = sys.argv[2]
    db = sys.argv[3]
    dst = sys.argv[4]
    main(folder, img, db, dst)

```

From bottom to top, we have the `main` controlling the main loop, `processTar` to controlling the processing for
individual data sets, packages into tarballs and `process` doing the actual work, processing a single frame.

## The Main loop
The main loop iterates over each tarball (data set) we have. Also doing some runtime estimations, cleaning up after
potentially failing workers and doing basic check pointing.

```python
def main(folder, img, db, dst):
    times = []
    ctr = 0
    amnt = len(os.listdir(folder))
    for tarball in os.listdir(folder):
        ctr += 1
        print("\n\n\nProcessing {}: {}/{} ({}%) \n\n\n\n".format(tarball, ctr, amnt, (float(ctr)/amnt)*100))
        if os.path.exists(os.path.join(dst,"hist_{}_{}.sqlite".format(tarball,db))):
            print("{} already done. skipping.".format(tarball))
            continue
        start = datetime.now()
        try:
            processTar(os.path.join(folder,tarball))
        except:
            print("{} failed. skipping.".format(tarball))
            shutil.rmtree(os.path.join("/dev/shm", tarball))
        else:
            shutil.copyfile(
                "hist_{}_{}.sqlite".format(tarball,db),
                os.path.join(dst,"hist_{}_{}.sqlite".format(tarball,db))
            )
            end = datetime.now()
            times.append(end-start)
            print("\n\n{} took {} - AVG {} - TTE {}, ETA {}".format(
                tarball,
                end-start,
                sum(times, timedelta(0)) / len(times),
                sum(times, timedelta(0)),
                (sum(times, timedelta(0)) / len(times)) * (amnt - ctr)
            ))
```

The check pointing is more important than you might think. If the program crashes for some reason, it can more or less continue where it
left off without loosing too much work. Processing on data set takes about 2.5 hours. Output data is created by the worker after it is done
with the data set. By checking for the existence of output data, already processes data sets are skipped. There we will at most loose 2.5 hours
in case of a crash, which is acceptable.

## The controlling function
The function `processTar` is called sequentially by the main loop. It prepares the data set by extracting the tarball to `/dev/shm`, which is a
shared-memory folder inside the main memory. Then calling the processing function for each individual frame. The results are parsed and saved to
a sqlite database.

```python
def processTar(tarball):
    if not tarball.endswith(".tar"):
        return

    images = dict()

    import tarfile
    f = tarfile.open(tarball, 'r')
    dirn = "/dev/shm/{}".format(os.path.basename(tarball))
    try:
        os.mkdir(dirn)

        f.extractall(path=dirn)
    except FileExistsError:
        pass

    if DEBUG:
        for img in os.listdir(dirn):
            start=datetime.now()
            process(os.path.join(dirn,img))
            print("{}: Time: {}".format(os.path.basename(img), datetime.now()-start))
    else:
        files = os.listdir(dirn)
        Parallel(n_jobs=150, backend="multiprocessing", batch_size=10, verbose=10)(
            delayed(process)(os.path.join(dirn,img)) for img in files)

    for f in os.listdir(dirn):
        if f.endswith(".txt"):
            with open(os.path.join(dirn, f), 'r') as fd:
                lines = fd.readlines()
                color_ints = [int(n.strip()) for n in lines]
                images.update({f[:-4]: color_ints})

    connection = sqlite3.connect("hist_{}_{}.sqlite".format(os.path.basename(tarball),db), check_same_thread = False)
    c = connection.cursor()
    c.execute("CREATE TABLE images (imgname text)")
    connection.commit()
    for i in range(CLUSTERS):
        c.execute("ALTER TABLE images ADD COLUMN color{} integer".format(i))

    connection.commit()
    for key, value in images.items():
        c.execute("INSERT INTO images VALUES (\"{}\",{})".format(key,
                                                                 ",".join([str(i) for i in value])))
    connection.commit()
    connection.close()

    shutil.rmtree(dirn)
```

The `process` function is called in a parallel fashion. The machine where this may or may not run, has 160 threads.
Some of the library-functions used in there, seam to facilitate multithreading, so I'm starting 150 workers allowing a slight overhead.

For each frame a `.txt` file is created containing the result. The results are now read and saved to a sqlite database
(to get some structure to the data). I've done it this way because there is otherwise no shared memory between the
workers, because each worker is a separate process. Processes are required because of problems with the Global Interpreter Lock (GIL)
being acquired or not released, making threads run more or less mutual exclusive, thus making them run basically sequentially.

## The frame Processing
This part has also some helper functions to prepare data and to mute the quite verbose library.

First again some form of check pointing, if the `.txt` exists the frame is considered done saving valuable time on recalculating.
If this works and the data in `/dev/shm` wasn't deleted, we lose virtually no time when something crashes. But if the data in
`/dev/shm` is lost than we are back to about 2.5 h lost.

Then after reading the image, something special happens, with a split of 14:4 the worker decides whether to use the CPU or GPU
for calculation. GPU calculating is roughly twice as fast as CPU calculation. But only about 22 processes per GPU can be run
concurrently. That's why the wired split.

The GPU worker then decides which GPU to use and copies the image data to the GPUs memory. Then KMeans is executed and the color
ratios are calculated. APIs for GPU and CPU are essentially the same.

Last but not least, the RGB values are stored as a single 32 integer, by concatenating the bits.

```python
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

@contextlib.contextmanager
def nostderr():
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stderr

def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c,column in enumerate(df):
      pdf[str(c)] = df[column]
    return pdf

def process(filen):
    try:
        if not filen.endswith("jpg") or os.path.exists("{}.txt".format(filen)):
            return

        img = cv2.imread(filen)
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        worker = multiprocessing.current_process()._identity[0]
        try:
            if worker%13 > 3:
                raise RuntimeError()
            logging.info("Starting CUDA-KMeans in Worker {} on Card {} for file {}".format(worker, worker%2, os.path.basename(filen)))
            cuda.select_device(worker%2)
            kmeans = KMeans(n_clusters=CLUSTERS, n_init=5, verbose=0)
            img_cuda = np2cudf(np.float32(img))
            with nostdout(), nostderr():
                kmeans.fit(img_cuda)
            center = cp.asarray(kmeans.cluster_centers_.values)
            label = cp.asarray(kmeans.labels_.data.mem)

            numLabels = cp.arange(0, CLUSTERS + 1)
            (hist, he) = cp.histogram(label, bins=numLabels)
            hist = hist.astype("float")
            hist /= hist.sum()
            colors = (center[(-hist).argsort()]).get()

            del kmeans
            del img_cuda
            del center
            del label
            del numLabels
            del hist
            del he
        except (RuntimeError, CudaAPIError):
            logging.info("Starting SKLearn-KMeans in Worker {} on CPU for file {}".format(worker, os.path.basename(filen)))
            kmeans = KMeans(n_clusters=CLUSTERS, n_init=5, precompute_distances=True, n_jobs=1, verbose=0)
            kmeans.use_gpu = False
            with nostdout(), nostderr():
                kmeans.fit(img)
            center = kmeans.cluster_centers_
            label = kmeans.labels_

            del kmeans

            numLabels = np.arange(0, CLUSTERS + 1)
            (hist, _) = np.histogram(label, bins=numLabels)
            hist = hist.astype("float")
            hist /= hist.sum()
            colors = center[(-hist).argsort()]

        logging.debug("{}: {}".format(filen, str(colors)))
        with open("{}.txt".format(filen), 'w') as fd:
            for i in range(CLUSTERS):
                col = int(colors[i][2]) << 16 | int(colors[i][1]) << 8 | int(colors[i][0])
                assert col <= 2**24
                fd.write("{}\n".format(str(col)))

    except Exception as e:
        logging.error(str(e))
        pass

```

# Setting up the Environment
Keen-eyed readers might have noticed, that I've omitted the imports in the code snippets above.
This was partially on purpose, because they don't matter that much when explaining what the
functions do. But because I know, how annoying it is to figure out the required imports and
by extent required libraries, I will explain in this section the required imports and packages.

My plain import section looks like this (BTW you also find this in the github-repo)
```python
import sys, os
import cv2
from pai4sk.cluster import KMeans
import cudf, cuml
import pandas as pd
import numpy as np
import sqlite3
from joblib import Parallel, delayed
import logging
from typing import *
import contextlib
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError
import multiprocessing
import cupy as cp
import shutil
from datetime import datetime
from datetime import timedelta
import time
```

`time`, `datetime`, `shutil`, `typing`, `sys`, `os`, `multiprocessing`, `sqlite3`, `logging` and
`contextlib` should be standard in any python installation.

The following packages should be easily installable (your millage may vary) with a package manager and a working
internet connection (which I hadn't, requiring a proxy, but that's out of scope)

| import           | pypi installtion            | conda installation                    | apt installtion (ubuntu)           |
|:-----------------|:----------------------------|:--------------------------------------|:-----------------------------------|
| numpy *          | `pip install numpy`         | `conda install -c conda-forge numpy`  | `apt install python3-numpy/focal`  |
| pandas **        | `pip install pandas`        | `conda install -c conda-forge pandas` | `apt install python3-pandas/focal` |
| joblib           | `pip install joblib`        | `conda install -c conda-forge joblib` | `apt install python3-joblib/focal` |
| cv2 *            | `pip install opencv-python` | `conda install -c conda-forge opencv` | `apt install python3-opencv/focal` |
| cupy             | `pip install cupy-cuda102`  | `conda install -c conda-forge cupy`   | --- |
| cudf **          | ---                         | See [https://rapids.ai/start.html](https://rapids.ai/start.html)    | ---                                |
| cuml **          | ---                         | See [https://rapids.ai/start.html](https://rapids.ai/start.html)    | ---                                |
| numba **         | `pip install numba`         | `conda install numba`                 | `apt install python3-numba/focal`  |

\* quite a hastle on non-x86 w/o conda, look [here](numpy_opencv_from_source.html) for installtion from source \\
\** are included as requirement  with powerAI stuff blow.

## PowerAI installation

_more in-depth information [here](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.7.0/navigation/wmlce_install.htm)_

As you might have realised by now, that the project is more tailored for IBMs Power-Architecture. Even if
there is no specific reason, why this shouldn't also work on x86-hardware, your mileage may vary.

To install the IBM Watson Machine Learning framework/PowerAI Library/SnapML Library (I'm honestly not
sure about the name, the name changes with each part of the documentation. Maybe IBM is just bad in
naming things.), you first need to install Anaconda, either via your systems package manager or
with the download from [anaconda.com](https://www.anaconda.com/products/individual)

{% include warning.html content="Because Anaconda automatically hooks into your shells startup
script (e.g. `.bashrc`), it tents to interfere with some desktop environments like KDE. <br/><br/>
If you experience problem with your desktop environment, not starting up anymore after Anaconda
installation, than try disabling it in your shells startup script, on console (VT1-7)." %}

With Anaconda installed now do the following

```bash
conda config --prepend channels \
https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
```

to add IBMs software repository. Now can create and active the environment we are going to use

```bash
conda create --name KMeans python=3.6
conda activate KMeans
```

Please note that on time of writing, only python 3.6 is going to work with the CUDA stuff.

Now we install the required packages to the environment

```bash
# IBM stuff
export IBM_POWERAI_LICENSE_ACCEPT=yes
conda install powerai
conda install powerai-rapids
# other stuff required
conda install -c conda-forge opencv
conda install -c conda-forge joblib
conda install -c conda-forge cupy
```

{% include note.html content="According to the IBM docs, `cudf` and `cuml` (which are definitely required)
are not available on x86. I assume this is meant in regard to their repository and not in general.
But your mileage my vary" %}

Everything should be set now to crunch some numbers!
