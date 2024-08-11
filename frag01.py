import numpy as np
import pandas as pd
import cv2
import svgwrite
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Function to check collinearity of three points
def are_collinear(p1, p2, p3):
    return abs((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) < 1e-6

# Function to combine paths based on collinearity for regular polygons
def combine_paths_for_polygon(paths):
    combined_paths = []
    while paths:
        base_path = paths.pop(0)
        i = 0
        while i < len(paths):
            extended = False
            for pt in [base_path[0], base_path[-1]]:
                for j in range(len(paths)):
                    if are_collinear(pt, *paths[j][:2]) or are_collinear(pt, *paths[j][-2:]):
                        if are_collinear(paths[j][0], paths[j][-1], pt):
                            if pt == base_path[0]:
                                base_path = paths.pop(j) + base_path
                            else:
                                base_path.extend(paths.pop(j))
                            extended = True
                            break
                if extended:
                    break
            if not extended:
                i += 1
        combined_paths.append(base_path)
    return combined_paths

# Function to check if paths form a closed shape
def is_closed_shape(path):
    return path[0] == path[-1]

# Function to create shapes from paths
def create_shapes_from_paths(paths):
    shapes = []
    for path in paths:
        if is_closed_shape(path):
            shapes.append(Polygon(path))
        else:
            shapes.append(LineString(path))
    return shapes

# Visualizing shapes
def visualize_shapes(shapes):
    fig, ax = plt.subplots()
    for shape in shapes:
        if isinstance(shape, Polygon):
            x, y = shape.exterior.xy
            ax.plot(x, y)
        elif isinstance(shape, LineString):
            x, y = shape.xy
            ax.plot(x, y)
    plt.show()

# Shape Identification Functions
def is_straight_line(XY):
    if len(XY) < 2:
        return False
    [vx, vy, x, y] = cv2.fitLine(XY.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    return np.allclose(XY[:, 1], vy/vx * (XY[:, 0] - x) + y, atol=1.0)

def is_circle_or_ellipse(XY):
    if len(XY) < 5:
        return False
    max_x = int(XY[:, 0].max())
    max_y = int(XY[:, 1].max())
    img = np.zeros((max_y + 1, max_x + 1), np.uint8)

    for x, y in XY:
        img[int(y), int(x)] = 255
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    
    return circles is not None

def is_rectangle_or_rounded_rectangle(XY):
    if len(XY) < 4:
        return False
    XY = XY.astype(np.float32)
    epsilon = 0.02 * cv2.arcLength(XY, True)
    approx = cv2.approxPolyDP(XY, epsilon, True)
    return len(approx) == 4

def is_regular_polygon(XY):
    if len(XY) < 5:
        return False
    
    XY = XY.astype(np.float32)
    epsilon = 0.02 * cv2.arcLength(XY, True)
    approx = cv2.approxPolyDP(XY, epsilon, True)
    
    if len(approx) < 5:
        return False
    
    # Calculate angles between consecutive edges
    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    angles = []
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        p3 = approx[(i + 2) % len(approx)][0]

        v1 = p2 - p1
        v2 = p3 - p2

        angle = angle_between(v1, v2)
        angles.append(angle)
    
    # Check if all angles are approximately equal
    mean_angle = np.mean(angles)
    angle_deviation = np.std(angles)
    
    return angle_deviation < 0.1  # Adjust this threshold as needed

def is_star_shape(XY):
    if len(XY) < 10:
        return False
    moments = cv2.moments(XY.astype(np.float32))
    return moments['m00'] != 0

def adjust_polygon(points, margin=10):
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    width = max_x - min_x
    height = max_y - min_y

    scale_x = (width - 2 * margin) / width
    scale_y = (height - 2 * margin) / height

    new_points = []
    for x, y in points:
        new_x = min_x + margin + scale_x * (x - min_x)
        new_y = min_y + margin + scale_y * (y - min_y)
        new_points.append((new_x, new_y))

    return new_points

import svgwrite
import numpy as np
from scipy.interpolate import splprep, splev

def smoothen_with_spline(XY):
    if XY.shape[0] < 3:
        print("Not enough points to perform spline smoothing.")
        return XY
    try:
        tck, u = splprep([XY[:, 0], XY[:, 1]], s=2)
        unew = np.linspace(0, 1, 100)
        out = splev(unew, tck)
        smoothed_XY = np.vstack(out).T
        return smoothed_XY
    except Exception as e:
        print(f"Error in spline smoothing: {e}")
        return XY

def smoothen_with_bezier(XY):
    if isinstance(XY, list):  # Convert XY to a NumPy array if it's a list
        XY = np.array(XY)
    if XY.shape[0] < 3:
        print("Not enough points to perform spline smoothing.")
        return XY
    try:
        path_data = f'M {XY[0, 0]} {XY[0, 1]} '  # Start the path with 'M' command
        for i in range(1, len(XY) - 1, 3):
            if i + 2 < len(XY):
                path_data += f'C {XY[i, 0]} {XY[i, 1]} {XY[i + 1, 0]} {XY[i + 1, 1]} {XY[i + 2, 0]} {XY[i + 2, 1]} '
            else:
                path_data += f'L {XY[i, 0]} {XY[i, 1]} '
        path = svgwrite.path.Path(d=path_data.strip(), stroke='black', fill='none')
        return path
    except Exception as e:
        print(f"Error in Bezier smoothing: {e}")
        return None

def constrain_points_within_circle(points, center, radius):
    constrained_points = []
    for x, y in points:
        dx, dy = x - center[0], y - center[1]
        dist = np.sqrt(dx*2 + dy*2)
        if dist > radius:
            scale = radius / dist
            x, y = center[0] + dx * scale, center[1] + dy * scale
        constrained_points.append((x, y))
    return constrained_points

import numpy as np
import pandas as pd
import cv2
import svgwrite
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Function to check collinearity of three points
def are_collinear(p1, p2, p3):
    return abs((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) < 1e-6

# Function to combine paths based on collinearity for regular polygons
def combine_paths_for_polygon(paths):
    combined_paths = []
    while paths:
        base_path = paths.pop(0)
        i = 0
        while i < len(paths):
            extended = False
            for pt in [base_path[0], base_path[-1]]:
                for j in range(len(paths)):
                    if are_collinear(pt, *paths[j][:2]) or are_collinear(pt, *paths[j][-2:]):
                        if are_collinear(paths[j][0], paths[j][-1], pt):
                            if pt == base_path[0]:
                                base_path = paths.pop(j) + base_path
                            else:
                                base_path.extend(paths.pop(j))
                            extended = True
                            break
                if extended:
                    break
            if not extended:
                i += 1
        combined_paths.append(base_path)
    return combined_paths

# Function to check if paths form a closed shape
def is_closed_shape(path):
    return path[0] == path[-1]

# Function to create shapes from paths
def create_shapes_from_paths(paths):
    shapes = []
    for path in paths:
        if is_closed_shape(path):
            shapes.append(Polygon(path))
        else:
            shapes.append(LineString(path))
    return shapes

# Visualizing shapes
def visualize_shapes(shapes):
    fig, ax = plt.subplots()
    for shape in shapes:
        if isinstance(shape, Polygon):
            x, y = shape.exterior.xy
            ax.plot(x, y)
        elif isinstance(shape, LineString):
            x, y = shape.xy
            ax.plot(x, y)
    plt.show()

# Shape Identification Functions
def is_straight_line(XY):
    if len(XY) < 2:
        return False
    [vx, vy, x, y] = cv2.fitLine(XY.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    return np.allclose(XY[:, 1], vy/vx * (XY[:, 0] - x) + y, atol=1.0)

def is_circle_or_ellipse(XY):
    if len(XY) < 5:
        return False
    max_x = int(XY[:, 0].max())
    max_y = int(XY[:, 1].max())
    img = np.zeros((max_y + 1, max_x + 1), np.uint8)

    for x, y in XY:
        img[int(y), int(x)] = 255
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    
    return circles is not None

def is_rectangle_or_rounded_rectangle(XY):
    if len(XY) < 4:
        return False
    XY = XY.astype(np.float32)
    epsilon = 0.02 * cv2.arcLength(XY, True)
    approx = cv2.approxPolyDP(XY, epsilon, True)
    return len(approx) == 4

def is_regular_polygon(XY):
    if len(XY) < 5:
        return False
    
    XY = XY.astype(np.float32)
    epsilon = 0.02 * cv2.arcLength(XY, True)
    approx = cv2.approxPolyDP(XY, epsilon, True)
    
    if len(approx) < 5:
        return False
    
    # Calculate angles between consecutive edges
    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    angles = []
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        p3 = approx[(i + 2) % len(approx)][0]

        v1 = p2 - p1
        v2 = p3 - p2

        angle = angle_between(v1, v2)
        angles.append(angle)
    
    # Check if all angles are approximately equal
    mean_angle = np.mean(angles)
    angle_deviation = np.std(angles)
    
    return angle_deviation < 0.1  # Adjust this threshold as needed

def is_star_shape(XY):
    if len(XY) < 10:
        return False
    moments = cv2.moments(XY.astype(np.float32))
    return moments['m00'] != 0

def adjust_polygon(points, margin=10):
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    width = max_x - min_x
    height = max_y - min_y

    scale_x = (width - 2 * margin) / width
    scale_y = (height - 2 * margin) / height

    new_points = []
    for x, y in points:
        new_x = min_x + margin + scale_x * (x - min_x)
        new_y = min_y + margin + scale_y * (y - min_y)
        new_points.append((new_x, new_y))

    return new_points

import svgwrite
import numpy as np
from scipy.interpolate import splprep, splev

def smoothen_with_spline(XY):
    if XY.shape[0] < 3:
        print("Not enough points to perform spline smoothing.")
        return XY
    try:
        tck, u = splprep([XY[:, 0], XY[:, 1]], s=2)
        unew = np.linspace(0, 1, 100)
        out = splev(unew, tck)
        smoothed_XY = np.vstack(out).T
        return smoothed_XY
    except Exception as e:
        print(f"Error in spline smoothing: {e}")
        return XY

def smoothen_with_bezier(XY):
    if isinstance(XY, list):  # Convert XY to a NumPy array if it's a list
        XY = np.array(XY)
    if XY.shape[0] < 3:
        print("Not enough points to perform spline smoothing.")
        return XY
    try:
        path_data = f'M {XY[0, 0]} {XY[0, 1]} '  # Start the path with 'M' command
        for i in range(1, len(XY) - 1, 3):
            if i + 2 < len(XY):
                path_data += f'C {XY[i, 0]} {XY[i, 1]} {XY[i + 1, 0]} {XY[i + 1, 1]} {XY[i + 2, 0]} {XY[i + 2, 1]} '
            else:
                path_data += f'L {XY[i, 0]} {XY[i, 1]} '
        path = svgwrite.path.Path(d=path_data.strip(), stroke='black', fill='none')
        return path
    except Exception as e:
        print(f"Error in Bezier smoothing: {e}")
        return None

def constrain_points_within_circle(points, center, radius):
    constrained_points = []
    for x, y in points:
        dx, dy = x - center[0], y - center[1]
        dist = np.sqrt(dx*2 + dy*2)
        if dist > radius:
            scale = radius / dist
            x, y = center[0] + dx * scale, center[1] + dy * scale
        constrained_points.append((x, y))
    return constrained_points
def compress_polyline(XY, epsilon=2.0):
    XY = XY.astype(np.float32)
    compressed = cv2.approxPolyDP(XY, epsilon, False)
    return compressed.reshape(-1, 2)
def constrain_points_within_circle(points, center, radius):
    constrained_points = []
    for x, y in points:
        dx, dy = x - center[0], y - center[1]
        dist = np.sqrt(dx**2 + dy**2)  # Fix: Use **2 for correct distance calculation
        if dist > radius:
            scale = radius / dist
            x, y = center[0] + dx * scale, center[1] + dy * scale
        constrained_points.append((x, y))
    return constrained_points

def straighten_and_save_to_svg(shapes, svg_path, center, radius, stroke_width=2):
    dwg = svgwrite.Drawing(svg_path, profile='tiny')

    for shape in shapes:
        if isinstance(shape, LineString):
            XY = np.array(shape.xy).T
        elif isinstance(shape, Polygon):
            XY = np.array(shape.exterior.xy).T
        else:
            continue

        if len(XY) == 0:
            continue
        XY = compress_polyline(XY)
        XY = constrain_points_within_circle(XY, center, radius)
        XY = np.array(XY)  # Ensure XY is a NumPy array

        if len(XY) >= 2 and is_straight_line(XY):
            [vx, vy, x, y] = cv2.fitLine(XY.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((XY[:, 0].max() - x) * vy / vx) + y)
            dwg.add(dwg.line(start=(int(XY[:, 0].min()), int(lefty)), end=(int(XY[:, 0].max()), int(righty)), stroke=svgwrite.rgb(0, 0, 0, '%'), stroke_width=stroke_width))

        elif len(XY) >= 4 and is_rectangle_or_rounded_rectangle(XY):
            x, y, w, h = cv2.boundingRect(XY.astype(np.float32))
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            box = adjust_polygon(box)
            box = constrain_points_within_circle(box, center, radius)
            dwg.add(dwg.polygon([(int(p[0]), int(p[1])) for p in box], fill='none', stroke='black', stroke_width=stroke_width))

        elif len(XY) >= 5 and is_circle_or_ellipse(XY):
            center, axes, angle = cv2.fitEllipse(XY.astype(np.float32))
            cx, cy = int(center[0]), int(center[1])
            rx, ry = int(axes[0] / 2), int(axes[1] / 2)
            if abs(rx - ry) < 10:  # Treat as circle if radii are nearly equal
                if np.sqrt((cx - center[0])**2 + (cy - center[1])**2) + rx <= radius:  # Fix: Correct distance calculation
                    dwg.add(dwg.circle(center=(cx, cy), r=rx, stroke='black', fill='none', stroke_width=stroke_width))
            else:  # Treat as ellipse
                if np.sqrt((cx - center[0])**2 + (cy - center[1])**2) + max(rx, ry) <= radius:  # Fix: Correct distance calculation
                    dwg.add(dwg.ellipse(center=(cx, cy), r=(rx, ry), stroke='black', fill='none', stroke_width=stroke_width))

        elif is_regular_polygon(XY):
            # Calculate the centroid of the polygon
            polygon_center = np.mean(XY, axis=0)
            # Calculate the radius as the minimum distance from the centroid to any vertex
            min_distance = min(np.sqrt((polygon_center[0] - p[0])**2 + (polygon_center[1] - p[1])**2) for p in XY)  # Fix: Correct distance calculation
            # Draw a circle inside the polygon
            dwg.add(dwg.circle(center=(int(polygon_center[0]), int(polygon_center[1])), r=int(min_distance), stroke='black', fill='none', stroke_width=stroke_width))
        
        else:
            smoothed_XY = smoothen_with_spline(XY)
            smoothed_XY = constrain_points_within_circle(smoothed_XY, center, radius)
            path = smoothen_with_bezier(smoothed_XY)
            if path is not None:
                path.stroke(width=stroke_width)  # Set the stroke width
                dwg.add(path)

    dwg.save()




# Symmetry and Bezier fitting functions
def find_symmetry(paths_XYs):
    symmetric_paths = []
    for XYs in paths_XYs:
        for XY in XYs:
            if len(XY.shape) == 1 or len(XY) == 0:
                continue
            if np.allclose(XY[:, 0], XY[:, 0][::-1]) or np.allclose(XY[:, 1], XY[:, 1][::-1]):
                symmetric_paths.append(XY)
    return symmetric_paths

def fit_bezier(XY):
    tck, u = splprep([XY[:, 0], XY[:, 1]], s=0)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

def fit_bezier_to_paths(paths_XYs):
    fitted_paths = []
    for XYs in paths_XYs:
        fitted_path = []
        for XY in XYs:
            if len(XY.shape) == 1 or len(XY) == 0:
                continue
            fitted_path.append(fit_bezier(XY))
        fitted_paths.append(fitted_path)
    return fitted_paths

# Plot function
def plot(path_XYs, title):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            if len(XY.shape) > 1 and XY.shape[1] == 2:  # Ensure XY is 2D
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.show()
def compress_polyline(XY, epsilon=2.0):
    XY = XY.astype(np.float32)
    compressed = cv2.approxPolyDP(XY, epsilon, False)
    return compressed.reshape(-1, 2)

# Save to CSV
def save_to_csv(paths_XYs, csv_path):
    with open(csv_path, 'w') as f:
        for path_index, XYs in enumerate(paths_XYs):
            if len(XYs) == 0:
                continue
            if isinstance(XYs, np.ndarray) and len(XYs.shape) == 2:  # Ensure XYs is a 2D array
                for point_index, point in enumerate(XYs):
                    f.write(f"{path_index},0,{point_index},{point[0]},{point[1]}\n")
            elif isinstance(XYs, list):  # Handle cases where XYs is a list of arrays
                for line_index, XY in enumerate(XYs):
                    if len(XY.shape) == 1 or len(XY) == 0:
                        continue
                    for point_index, point in enumerate(XY):
                        f.write(f"{path_index},{line_index},{point_index},{point[0]},{point[1]}\n")


# Symmetry and Bezier fitting functions
def find_symmetry(paths_XYs):
    symmetric_paths = []
    for XYs in paths_XYs:
        for XY in XYs:
            if len(XY.shape) == 1 or len(XY) == 0:
                continue
            if np.allclose(XY[:, 0], XY[:, 0][::-1]) or np.allclose(XY[:, 1], XY[:, 1][::-1]):
                symmetric_paths.append(XY)
    return symmetric_paths

def fit_bezier(XY):
    tck, u = splprep([XY[:, 0], XY[:, 1]], s=0)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

def fit_bezier_to_paths(paths_XYs):
    fitted_paths = []
    for XYs in paths_XYs:
        fitted_path = []
        for XY in XYs:
            if len(XY.shape) == 1 or len(XY) == 0:
                continue
            fitted_path.append(fit_bezier(XY))
        fitted_paths.append(fitted_path)
    return fitted_paths

# Plot function
def plot(path_XYs, title):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            if len(XY.shape) > 1 and XY.shape[1] == 2:  # Ensure XY is 2D
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.show()

# Save to CSV
def save_to_csv(paths_XYs, csv_path):
    with open(csv_path, 'w') as f:
        for path_index, XYs in enumerate(paths_XYs):
            if len(XYs) == 0:
                continue
            if isinstance(XYs, np.ndarray) and len(XYs.shape) == 2:  # Ensure XYs is a 2D array
                for point_index, point in enumerate(XYs):
                    f.write(f"{path_index},0,{point_index},{point[0]},{point[1]}\n")
            elif isinstance(XYs, list):  # Handle cases where XYs is a list of arrays
                for line_index, XY in enumerate(XYs):
                    if len(XY.shape) == 1 or len(XY) == 0:
                        continue
                    for point_index, point in enumerate(XY):
                        f.write(f"{path_index},{line_index},{point_index},{point[0]},{point[1]}\n")

# Main Usage
def main():
    csv_path = './problems/problems/frag1.csv'  # Path to your CSV file
    svg_path = './frag0_regularized.svg'  # Path to save the SVG file
    column_names = ['path_id', 'unused_col', 'x', 'y']
    df = pd.read_csv(csv_path, names=column_names, skiprows=1)

    # Group the dataframe by path_id and collect the coordinates
    paths = df.groupby('path_id')[['x', 'y']].apply(lambda g: list(zip(g['x'], g['y']))).tolist()
    combined_paths = combine_paths_for_polygon(paths)

    # Create shapes from combined paths
    shapes = create_shapes_from_paths(combined_paths)

    # Define the center and radius of the bounding circle
    center = (df['x'].mean(), df['y'].mean())
    radius = max(np.sqrt((df['x'] - center[0])*2 + (df['y'] - center[1])*2))

    # Save the regularized paths to an SVG file
    straighten_and_save_to_svg(shapes, svg_path, center, radius)

    # Visualize the shapes
    visualize_shapes(shapes)

    # Extract XY coordinates from shapes
    regularized_paths = [np.array(shape.exterior.xy).T if isinstance(shape, Polygon) else np.array(shape.xy).T for shape in shapes]

    # Plot the regularized shapes
    plot(regularized_paths, "Regularized Fragment 0")

    # Save the regularized paths to a CSV file
    output_csv_path = './regularized_frag0.csv'
    save_to_csv(regularized_paths, output_csv_path)

if __name__ == '__main__':
    main()
