# Vehicle_Speed_Detection-License_Plate_Recognition

## Project Description

This project is designed to detect the speed of vehicles and recognize license plates using computer vision techniques. The system uses video footage to capture vehicle movement, calculate speed, and identify license plate numbers in real-time.

## Features

- Detects vehicle speed from video footage.
- Recognizes and extracts license plate numbers.
- Real-time processing capabilities.
- Easy-to-use interface for monitoring and analysis.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/harshavangari/Vehicle_Speed_Detection-License_Plate_Recognition.git
    cd Vehicle_Speed_Detection-License_Plate_Recognition
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the video footage you want to analyze and place it in the appropriate directory (e.g., `videos/`).

2. Run the main script to start the detection process:
    ```bash
    python main.py --input videos/sample_video.mp4
    ```

3. The results will be displayed on the screen and saved to the output directory.

## Contributing

We welcome contributions to this project! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
