# HawkEyeTennis

## Tennis Match Analysis with Object Tracking and Stats

This project analyzes tennis match videos to track player and ball movements, detect court lines, and calculate detailed statistics like shot speeds, player speeds, and player actions. The output is a visually enhanced video with overlays, including bounding boxes, keypoints, mini-court projections, and player statistics.

## Features
- **Player Tracking**: Detects and tracks players using a YOLO-based model.
- **Ball Tracking**: Detects and tracks the ball using a custom-trained model.
- **Court Line Detection**: Identifies tennis court lines using a keypoints model.
- **Mini-Court Projection**: Projects player and ball movements onto a miniature court.
- **Player Statistics**: Calculates shot speeds, player speeds, and averages, and visualizes them on the video.
- **Video Output**: Generates a processed video with all visualizations and statistics.

## Project Structure
├── input\_videos/             # Directory containing input videos for analysis

├── output\_videos/            # Directory containing processed output videos

├── models/                   # Pre-trained models (YOLO, Court Line Detector, etc.)

├── tracker\_stubs/            # Stub files for pre-processed detections

├── utils/                    # Helper functions (video reading, saving, measurements)

├── court\_line\_detector/      # Logic for court line detection

├── mini\_court/               # Simplified mini-court representation for visualization

├── constants.py              # Global constants used throughout the project

├── main.py                   # Main script to execute the analysis

## 🚀 Features

1.  **Player Detection and Tracking**
    Detects players in video frames using a YOLO-based model and filters detections based on court keypoints.

2.  **Ball Detection and Interpolation**
    Detects ball positions and interpolates frames for smooth tracking.

3.  **Court Line Detection**
    Identifies keypoints of the tennis court for spatial reference and accurate mapping.

4.  **Mini-Court Visualization**
    Projects player and ball positions onto a scaled-down mini-court representation.

5.  **Statistics Calculation**
    Computes statistics such as:

    -   Number of shots per player
    -   Shot speeds (km/h)
    -   Player speeds (km/h)
6.  **Annotated Video Output**
    Generates output videos with annotations for player and ball positions, court lines, and statistics.

* * *

## 🛠 Requirements

To run this project, you will need the following:

-   Python 3.8 or higher
-   OpenCV
-   PyTorch
-   Pandas
-   Pre-trained YOLO models and court line detection models

Install dependencies using:


`pip install -r requirements.txt`

* * *

## 📖 Usage

1.  **Prepare Input Video**
    Place the video you want to analyze in the `input_videos/` directory.

2.  **Run the Main Script**
    Execute the main script to start the analysis:

    `python main.py`

3.  **Output Video**
    After processing, the output video with annotations will be saved in the `output_videos/`

## 🤝 Contribution

Feel free to fork this repository, submit issues, or create pull requests to contribute to the project.


* * *

Happy tracking! 🎾


