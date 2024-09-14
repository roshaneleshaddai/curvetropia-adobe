Curvetropia: Regularization and Symmetry Detection of Shapes
In this project, Curvetropia, we tackle the challenge of regularizing and identifying symmetrical shapes from a given set of fragmented SVG paths. This is a critical task in various fields, such as computer vision, graphic design, and CAD systems, where precise shape recognition and regularization are essential.

Project Overview
Curvetropia focuses on three main approaches to solve the problem of shape regularization and symmetry detection:

1. Model for Shape Identification
In this approach, we trained a model specifically designed to identify and classify various shapes. The model uses advanced image processing techniques and machine learning algorithms to detect different geometric shapes, such as circles, rectangles, and polygons. The primary goal is to automate the identification process, making it efficient and reliable.

2. Shape Identification and Regularization using OpenCV
This approach leverages the power of OpenCV, a popular open-source computer vision library, to identify and regularize shapes. The pipeline includes:
Shape Detection: Using edge detection and contour analysis to identify shapes.
Regularization: Applying techniques like polyline compression and smoothing with splines and Bezier curves to refine the shapes into their ideal geometric forms.

3. Identification and Regularization of Isolated Shapes
This method focuses on identifying isolated shapes within a cluttered environment. The approach combines clustering algorithms (like DBSCAN) and shape analysis to separate individual shapes from overlapping or connected ones. Once isolated, each shape undergoes a regularization process to achieve a more precise geometric representation.


## Contributors
- M. Eleshaddai Roshan
- G. Sreeja
