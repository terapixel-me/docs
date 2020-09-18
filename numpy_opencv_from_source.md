---
layout: page
title: Installing NumPy and OpenCV from source
nav_exclude: true
---
# Installing NumPy and OpenCV from source
If for some reason you decided not to use Anaconda and are on an non-x86 Architecture
where no prebuild packages for NumPy and OpenCV are available, than you can roughly
follow this to build these packages from source.

_Sidenote: I did this because at this point I hadn't opted for anaconda yet. But
I did the switch later, making the steps kinda pointless. Don't be like me, make
an educated decision before starting :)_

## NumPy
__Again there is no reason to do this on standard architectures with internet access available__

You first need to have Cython installed to need to install it from source. Download the
Cython sources from [PyPi](https://pypi.org/project/Cython/#files) and extract them. In
the sources-folder run this (assuming you have at least a standard C compiler installed)

```bash
python setup.py install
```

(please make sure `python` actually points to a Python 3 excutable this might not
always be the case)

This might require the permissions to write to some system-directories. If this is
the case there is a way around it, but bash-history is available anymore.

Now you need the NumPy sources

```bash
git clone https://github.com/numpy/numpy
```

And build NumPy inside the sources folder

```bash
python setup.py build_ext --inplace
```

You might need to specify where Cython is installed to. But these are things lost in
the shell :(

## OpenCV
OpenCV requires NumPy so you need to have it installed first.

```bash
git clone https://github.com/opencv/opencv.git
mkdir build
cd build
```

And now simply run

```bash
cmake ../
make
make install
```

By default this will try install system-wide. You can change this behavior but
again, thing lost in the shell.
