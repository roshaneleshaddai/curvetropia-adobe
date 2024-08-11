import cv2
import numpy as np

def smooth_contours(image, contours):
    """Smooth the contours by applying a Gaussian blur."""
    # Create a mask for drawing the contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Apply Gaussian blur to the mask
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Combine the blurred mask with the original image
    smoothed_image = cv2.addWeighted(image, 0.5, blurred_mask, 0.5, 0)
    return smoothed_image

def detect_shapes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an image for results
    result_image = np.zeros_like(image)
    completed_image = image.copy()

    # Initialize counts
    symmetric_count = 0
    completed_count = 0
    regularized_count = 0

    for contour in contours:
        # Regularize the shape (approximate to a polygon)
        epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjusted epsilon for better approximation
        approx_curve = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the approximated (regularized) curve
        cv2.drawContours(result_image, [approx_curve], -1, (255, 255, 255), 2)
        regularized_count += 1

        # Check for symmetry by flipping the ROI
        x, y, w, h = cv2.boundingRect(approx_curve)
        roi = gray[y:y+h, x:x+w]

        # Ensure the ROI is not empty
        if roi.size == 0:
            continue

        roi_flipped_horizontally = cv2.flip(roi, 1)
        roi_flipped_vertically = cv2.flip(roi, 0)

        horizontal_symmetry = np.sum(roi == roi_flipped_horizontally) / roi.size
        vertical_symmetry = np.sum(roi == roi_flipped_vertically) / roi.size

        if horizontal_symmetry > 0.9 or vertical_symmetry > 0.9:
            symmetric_count += 1
            cv2.drawContours(result_image, [approx_curve], -1, (0, 255, 0), 2)

        # Create a mask for the current contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Complete incomplete curves using inpainting
        completed_image = cv2.inpaint(completed_image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        completed_count += 1

    # Smooth the contours in the result image
    smoothed_result_image = smooth_contours(result_image, [contour for contour in contours])

    # Print the number of symmetric, completed, and regularized shapes
    print(f"Number of symmetric shapes detected: {symmetric_count}")
    print(f"Number of completed shapes: {completed_count}")
    print(f"Number of regularized shapes: {regularized_count}")

    # Display the results
    cv2.imshow("Original Image", image)
    cv2.imshow("Edges", edges)
    cv2.imshow("Regularized and Smoothed Detection", smoothed_result_image)
    cv2.imshow("Completed Shapes", completed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    image_path = "./doodle.png"  # Replace with your image path
    detect_shapes(image_path)