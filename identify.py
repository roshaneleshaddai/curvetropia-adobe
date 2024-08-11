import numpy as np
import cv2
import svgwrite
import csv

# Read CSV and process paths
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)

    return path_XYs

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
    return len(approx) >= 5

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

def straighten_and_save_to_svg_and_csv(path_XYs, svg_path, csv_output_path):
    dwg = svgwrite.Drawing(svg_path, profile='tiny')
    csv_data = []

    for path_id, XYs in enumerate(path_XYs, start=1):
        for XY in XYs:
            if len(XY) == 0:
                continue
            if len(XY) >= 2 and is_straight_line(XY):
                [vx, vy, x, y] = cv2.fitLine(XY.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((XY[:, 0].max() - x) * vy / vx) + y)
                dwg.add(dwg.line(start=(int(XY[:, 0].min()), int(lefty)), end=(int(XY[:, 0].max()), int(righty)), stroke=svgwrite.rgb(0, 0, 0, '%')))
                
                # Add to CSV data
                for point in XY:
                    csv_data.append([path_id, 0, format(point[0], '.2e'), format(point[1], '.2e')])
                    
            elif len(XY) >= 5 and is_circle_or_ellipse(XY):
                ellipse = cv2.fitEllipse(XY.astype(np.float32))
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                radius = int((ellipse[1][0] + ellipse[1][1]) / 4)  # Average the axes to get a radius for a circle
                dwg.add(dwg.circle(center=center, r=radius, fill='none', stroke='black'))
                
                # Add to CSV data
                for point in XY:
                    csv_data.append([path_id, 0, format(point[0], '.2e'), format(point[1], '.2e')])
                    
            elif len(XY) >= 4 and is_rectangle_or_rounded_rectangle(XY):
                x, y, w, h = cv2.boundingRect(XY.astype(np.float32))
                box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                box = adjust_polygon(box)
                dwg.add(dwg.polygon([(int(p[0]), int(p[1])) for p in box], fill='none', stroke='black'))
                
                # Add to CSV data
                for point in box:
                    csv_data.append([path_id, 0, format(point[0], '.2e'), format(point[1], '.2e')])
                    
            elif is_regular_polygon(XY):
                epsilon = 0.02 * cv2.arcLength(XY.astype(np.float32), True)
                approx = cv2.approxPolyDP(XY.astype(np.float32), epsilon, True)
                if len(approx) >= 3:
                    points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                    points = adjust_polygon(points)
                    dwg.add(dwg.polygon(points, fill='none', stroke='black'))
                    
                    # Add to CSV data
                    for point in points:
                        csv_data.append([path_id, 0, format(point[0], '.2e'), format(point[1], '.2e')])
                    
            elif is_star_shape(XY):
                points = [(int(p[0]), int(p[1])) for p in XY]
                points = adjust_polygon(points)
                dwg.add(dwg.polygon(points, fill='none', stroke='black'))
                
                # Add to CSV data
                for point in points:
                    csv_data.append([path_id, 0, format(point[0], '.2e'), format(point[1], '.2e')])
                    
            else:
                dwg.add(dwg.polyline([(int(p[0]), int(p[1])) for p in XY], fill='none', stroke='black'))
                
                # Add to CSV data
                for point in XY:
                    csv_data.append([path_id, 0, format(point[0], '.2e'), format(point[1], '.2e')])

    dwg.save()
    
    # Save CSV file
    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Path ID', 'Zero', 'X', 'Y'])
        writer.writerows(csv_data)

# Usage
csv_path = './problems/problems/isolated.csv'  
svg_path = './isolated_sol.svg'  
csv_output_path = './isolated_output.csv'
paths = read_csv(csv_path)

straighten_and_save_to_svg_and_csv(paths, svg_path, csv_output_path)
print(f"SVG file saved to {svg_path}")
print(f"CSV file saved to {csv_output_path}")

# # Code to read and plot the CSV file
# import matplotlib.pyplot as plt

# def read_csv_for_plotting(csv_path):
#     data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
#     path_XYs = []

#     for i in np.unique(data[:, 0]):
#         path_data = data[data[:, 0] == i][:, 2:]
#         path_XYs.append(path_data)
    
#     return path_XYs


# def plot_paths(path_XYs):
#     colours = ['b']
#     fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    
#     for i, XY in enumerate(path_XYs):
#         c = colours[i % len(colours)]
#         ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    
#     ax.set_aspect('equal')
#     plt.show()

# csv_output_path = './isolated_output.csv'
# paths = read_csv_for_plotting(csv_output_path)
# plot_paths(paths)