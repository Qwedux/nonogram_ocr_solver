import cv2
import numpy as np
import os

# Method 1: Set environment variables for Qt scaling (recommended)
# These should ideally be set before importing cv2, or in your shell profile
os.environ['QT_SCALE_FACTOR'] = '2'  # Scale factor (2 for 200%, 3 for 300%)
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

class EdgeDetectorApp:
    def __init__(self, image_path='image.png', scale_factor=2):
        self.scale_factor = scale_factor
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            # Create a sample image if file doesn't exist
            print(f"Could not load {image_path}, creating sample image...")
            self.image = self.create_sample_image()
        
        # Convert to grayscale for edge detection
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Initial threshold values
        self.low_threshold = 50
        self.high_threshold = 120
        
        self.setup_window()
        
    def create_sample_image(self):
        """Create a sample image with some shapes for testing"""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 150), (255, 255, 255), -1)
        cv2.circle(img, (400, 100), 75, (128, 128, 128), -1)
        cv2.ellipse(img, (300, 300), (100, 50), 45, 0, 360, (200, 200, 200), -1)
        return img
    
    def setup_window(self):
        """Setup the OpenCV window with scaled trackbars"""
        window_name = 'Interactive Edges'
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Method 2: Resize the window to be larger
        # This makes the trackbars more usable on high-DPI displays
        cv2.resizeWindow(window_name, 800 * self.scale_factor, 600 * self.scale_factor)
        
        # Create trackbars
        cv2.createTrackbar('Low Threshold', window_name, self.low_threshold, 255, self.update_edges)
        cv2.createTrackbar('High Threshold', window_name, self.high_threshold, 255, self.update_edges)
        
        # Initial edge detection
        self.update_edges(0)
        
    def update_edges(self, val):
        """Callback function for trackbar changes"""
        window_name = 'Interactive Edges'
        
        # Get current trackbar values
        self.low_threshold = cv2.getTrackbarPos('Low Threshold', window_name)
        self.high_threshold = cv2.getTrackbarPos('High Threshold', window_name)
        
        # Ensure high threshold is always >= low threshold
        if self.high_threshold < self.low_threshold:
            self.high_threshold = self.low_threshold
            cv2.setTrackbarPos('High Threshold', window_name, self.high_threshold)
        
        # Apply Canny edge detection
        edges = cv2.Canny(self.gray, self.low_threshold, self.high_threshold)
        
        # Convert edges to 3-channel for display alongside original
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Create side-by-side comparison
        comparison = np.hstack((self.image, edges_colored))
        
        # Display the result
        cv2.imshow(window_name, comparison)
    
    def run(self):
        """Main application loop"""
        print("Edge Detection Controls:")
        print("- Use the sliders to adjust Canny edge detection thresholds")
        print("- Press 'q' to quit")
        print("- Press 's' to save the current edge image")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current edge detection result
                edges = cv2.Canny(self.gray, self.low_threshold, self.high_threshold)
                cv2.imwrite('edges_output.png', edges)
                print("Edge image saved as 'edges_output.png'")
        
        cv2.destroyAllWindows()

# Alternative Method 3: Using tkinter for better DPI handling
def create_tkinter_version():
    """Alternative implementation using tkinter for better DPI scaling"""
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        
        class TkinterEdgeDetector:
            def __init__(self, image_path='image.png'):
                self.root = tk.Tk()
                self.root.title("Edge Detection with Tkinter")
                
                # Load image
                self.image = cv2.imread(image_path)
                if self.image is None:
                    self.image = self.create_sample_image()
                
                self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
                # Create GUI elements
                self.setup_gui()
                
            def create_sample_image(self):
                img = np.zeros((400, 600, 3), dtype=np.uint8)
                cv2.rectangle(img, (50, 50), (200, 150), (255, 255, 255), -1)
                cv2.circle(img, (400, 100), 75, (128, 128, 128), -1)
                return img
            
            def setup_gui(self):
                # Control frame
                control_frame = ttk.Frame(self.root)
                control_frame.pack(pady=10)
                
                # Low threshold slider
                ttk.Label(control_frame, text="Low Threshold:").grid(row=0, column=0, padx=5)
                self.low_var = tk.IntVar(value=50)
                self.low_scale = ttk.Scale(control_frame, from_=0, to=255, 
                                         variable=self.low_var, command=self.update_image)
                self.low_scale.grid(row=0, column=1, padx=5)
                
                # High threshold slider  
                ttk.Label(control_frame, text="High Threshold:").grid(row=1, column=0, padx=5)
                self.high_var = tk.IntVar(value=120)
                self.high_scale = ttk.Scale(control_frame, from_=0, to=255,
                                          variable=self.high_var, command=self.update_image)
                self.high_scale.grid(row=1, column=1, padx=5)
                
                # Image label
                self.image_label = ttk.Label(self.root)
                self.image_label.pack(pady=10)
                
                # Initial update
                self.update_image()
            
            def update_image(self, *args):
                low = self.low_var.get()
                high = self.high_var.get()
                
                # Apply Canny edge detection
                edges = cv2.Canny(self.gray, low, high)
                
                # Convert for display
                edges_pil = Image.fromarray(edges)
                edges_pil = edges_pil.resize((600, 400))  # Resize for display
                
                self.photo = ImageTk.PhotoImage(edges_pil)
                self.image_label.config(image=self.photo)
            
            def run(self):
                self.root.mainloop()
        
        return TkinterEdgeDetector
    
    except ImportError:
        print("tkinter or PIL not available for alternative GUI")
        return None

if __name__ == "__main__":
    print("OpenCV Edge Detection with Scaling Solutions")
    print("=" * 50)
    
    # Method 1: Use environment variables (recommended)
    print("Starting OpenCV version with Qt scaling...")
    app = EdgeDetectorApp('image.png', scale_factor=2)
    app.run()
    
    # Uncomment below to try tkinter version instead
    # TkinterApp = create_tkinter_version()
    # if TkinterApp:
    #     print("Starting Tkinter version...")
    #     tk_app = TkinterApp('image.png')
    #     tk_app.run()