---
layout: page
title: Getting the source images
nav_order: 2
---
On the start-page of the docs, I said that this is a fan-art project for a YouTube channel.
So my initial source data are the 275 YouTube videos of the channel, split apart into its frames.

You may gather your images otherwise and organize them differently. The same principles apply. But the
fact, that I worked with video frames, hugely impacts how I manage the data set.

The videos are downloaded with youtube-dl.

```bash
youtube-dl -o "%(id)s.%(ext)s" -f mp4[height=1080,fps=60] https://www.youtube.com/playlist?list=UUIcgBZ9hEJxHv6r_jDYOMqg
```

This yield many errors, as many videos are not available as mp4 so it needed to be modified to

```bash
youtube-dl -o "%(id)s.%(ext)s" -f bestvideo[height=1080,fps=60] https://www.youtube.com/playlist?list=UUIcgBZ9hEJxHv6r_jDYOMqg
```

Than some videos weren't 60fps...

```bash
youtube-dl -o "%(id)s.%(ext)s" -f bestvideo[height=1080] https://www.youtube.com/playlist?list=UUIcgBZ9hEJxHv6r_jDYOMqg
```

And some not in a 16:9 aspect ratio

```bash
youtube-dl -o "%(id)s.%(ext)s" -f "bestvideo" -i https://www.youtube.com/watch?v=A_FaQBNzUAQ
youtube-dl -o "%(id)s.%(ext)s" -f "bestvideo" -i https://www.youtube.com/watch?v=rnQLOnnjpoc
```

And last but not least, there were age-restricted videos... So login to YouTube, install the
[cookies.txt](https://chrome.google.com/webstore/detail/cookiestxt/njabckikapfpffapmjgojcnbfjonfjfg)
Chrome extension (there is probably also something for Firefox, but I don't know. You will find it your self)
to create a curl compatible `cookies.txt`. And try again:

```bash
youtube-dl -o "%(id)s.%(ext)s" -f "bestvideo" -i --cookies cookies.txt https://www.youtube.com/watch?v=rnQLOnnjpoc
youtube-dl -o "%(id)s.%(ext)s" -f "bestvideo" -i --cookies cookies.txt https://www.youtube.com/watch?v=7takIh1nK0s
youtube-dl -o "%(id)s.%(ext)s" -f "bestvideo" -i --cookies cookies.txt https://www.youtube.com/watch?v=EeG7E8bCNu0
youtube-dl -o "%(id)s.%(ext)s" -f "bestvideo" -i --cookies cookies.txt https://www.youtube.com/watch?v=3gUWFxKG1vw
...
```

## Splitting the videos

Now that videos are downloaded, the frame need to be extracted.

```bash
ffmpeg -hide_banner -loglevel panic -stats -y -i input.mp4 -qscale:v 2 /frames/frame_%06d.jpg
```

This takes about the same amount of time as the videos plays. Doing this sequentially, is much too slow for my taste.
So I created this script:

```bash
#!/bin/bash -e

LD_LIBRARY_PATH=/scratch/bennet/ffmpeg-build/lib:/usr/local/cuda/lib64/

prefix="/scratch/bennet"
dest="/data/others/bennet/frames"

file="$prefix/video/$1"
filen="$1"

mkdir -p $prefix/frames/$filen
echo "$filen: spliting frames"
/scratch/bennet/ffmpeg-build/bin/ffmpeg -hide_banner -loglevel panic -stats -y -i $file -qscale:v 2 $prefix/frames/$filen/${filen}_%06d.jpg

pushd .


echo "$filen: packing frames"
cd $prefix/frames/$filen/
tar cf $filen.tar .

echo "$filen: moving to data"
mv -- $filen.tar $dest/

popd
rm -rf $prefix/frames/$filen/

echo "$filen: done"
```

Calling it like this on all the videos

```bash
ls videos/ | parallel -P 25% --progress ./split_single.sh {}
```

Here you see the data set structure I mentioned. Where the set of frame for video is packed to a tar archive (to reduce inodes required).
This specially crafted to may or may not run on a special HPC machine with a network attached large storage volume.

Doing some napkin math, assuming each video to be roughly 15 minutes long, with 45fps on average we get 40500 frames per video, which totals about
11137500 frames over 275 videos. This is the reason I need and want to save inodes.

The special machine in question may or may not is an IBM POWER8 machine with 2 CPUs, 40 Cores and 160 Threads (yes, eight-fold SMT!) and two NVIDIA Tesla P100 16GB SMX
with NVLink.
