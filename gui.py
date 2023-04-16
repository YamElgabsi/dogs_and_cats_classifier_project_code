import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

import matplotlib
import utils
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("500x700")

        # Create a label to display the selected image
        self.image_label = tk.Label(self.root)

        # Create a button to select an image
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)

        # Create a button to make a prediction
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_btn_listener)

        self.run_again = tk.Button(self.root, text="Select another", command=self.run)

        self.canvas = None
        # Initialize the file path
        self.file_path = None
        self.root.title("Dog, Horse, or Cat Classifier")
        self.image_label.pack()
        self.select_button.pack()
        self.root.mainloop()

    def run(self):
        self.run_again.pack_forget()
        self.predict_button.pack_forget()
        self.canvas._tkcanvas.pack_forget()
        self.select_image()


    def select_image(self):
        # Open a file dialog to select an image
        self.file_path = filedialog.askopenfilename(initialdir="/files", title="Select Image", filetypes=(
        ("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
        # Display the selected image
        self.display_image(self.file_path)
        self.select_button.pack_forget()
        self.predict_button.pack()

    def display_image(self, file_path):
        self.image_label.pack()
        # Load the image using PIL
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.Resampling.LANCZOS)

        # Convert the image to a PhotoImage object
        image = ImageTk.PhotoImage(image)

        # Update the image label with the new image
        self.image_label.config(image=image)
        self.image_label.image = image


    def predict_btn_listener(self):
        # Call the predict function with the file path
        fig = utils.get_prediction_graph(self.file_path)
        self.image_label.pack_forget()
        self.predict_button.pack_forget()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.image_label.image = None
        self.run_again.pack()

App()

