## AR_Plotting
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
---

## Overview

This project will focus on detecting a custom AR Tag (a form of fiducial marker), that is used for obtaining a point of
reference in the real world, such as in augmented reality applications. There are two aspects to using an AR Tag, namely
detection and tracking, both of which will be implemented in this project. The detection stage will involve finding
the AR Tag from a given image sequence while the tracking stage will involve keeping the tag in “view” throughout the
sequence and performing image processing operations based on the tag’s orientation and position (a.k.a. the pose).

## Softwares

* Recommended IDE: PyCharm 2021.2

## Libraries

* Numpy 1.21.2
* OpenCV 3.4.8.29

## Programming Languages

* Python 3.8.12

## License 

```
MIT License

Copyright (c) 2021 Rajan Pande

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```
## Bugs

* Flickering imposed image
* Improper AR cube placement

## Demo

Following demo is for Tag0.mp4

- [Superimposing Image:](https://youtu.be/RCyzip66vL8)

![ezgif com-gif-maker](https://github.com/rpande1996/AR_Plotting/blob/main/media/gif/testudo.gif)

- [AR Cube Plotting:](https://youtu.be/v349lcfAsPU)

![ezgif com-gif-maker](https://github.com/rpande1996/AR_Plotting/blob/main/media/gif/cube.gif)

## Build

```
git clone https://github.com/rpande1996/AR_Plotting
cd AR_Plotting/src
python AR_Cube_Plotting.py
```
Enter the video serial number you want to test:
1. Tag0.mp4
2. Tag1.mp4
3. Tag2.mp4
4. multipleTags.mp4