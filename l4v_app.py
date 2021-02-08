"""
Lookout for Vision Demo Application
                ___________
            .-=d88888888888b=-.
        .:d8888pr"|\\|/-\\|'rq8888b.
      ,:d8888P^//\\-\\/_\\ /_\\/^q888/b.
    ,;d88888/~-/ .-~  _~-. |/-q88888b,
   //8888887-\\ _/    (#)  \\-\\/Y88888b\\
   \\8888888|// T      `    Y _/|888888 o
    \\q88888|- \\l           !\\_/|88888p/
     'q8888l\\-//\\         / /\\|!8888P'
       'q888\\/-| "-,___.-^\\/-\\/888P'
         `=88\\./-/|/ |-/!\\/-!/88='
            ^^"-------------"^

Instructions:
    0) Install dependencies
        pip install -r requirements.txt
    1) Setup AWS environment variables
    2) Download extra images
        aws s3 cp s3://circuitboarddataset/circuit_board/extra_images/ extra_images/ --recursive
    3) Run application

For more information, please visit:
https://aws.amazon.com/lookout-for-vision/
"""

import argparse
import logging
import os
import sys

from glob import glob
from random import choice

from pandas import DataFrame

import boto3
import cv2

# Constants
DEFAULT_WINDOW_NAME = 'Lookout for Vision Demo'
DEFAULT_TEXT_PADDING = 50
DEFAULT_WAITING_PERIOD = 5
CORRECT_PREDICTION_LABEL = "CORRECT"
INCORRECT_PREDICTION_LABEL = "INCORRECT"

class InvalidImageFormatError(Exception):
    """Raised when using an invalid image format
    Lookout for Vision only supports JPEG and PNG"""
    def __init__(self, image_format, message="Invalid image format"):
        self.image_format = image_format
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}: {self.image_format}'


def set_content_type(image):
    """Sets the content type of an image"""
    img_ext = os.path.splitext(image)[-1].lower()
    if img_ext in [".jpeg", ".jpg"]:
        return "image/jpeg"
    if img_ext in [".png"]:
        return "image/png"
    raise InvalidImageFormatError(img_ext)


def resize_with_aspect_ratio(image, target_width=None, target_height=None, inter=cv2.INTER_AREA):
    """Resizes an image while maintaining the aspect ratio"""

    # Get current image dimensions
    (current_height, current_width) = image.shape[:2]

    # Just return the original image if no target dimensions are provided
    if target_width and target_height:
        return image

    if target_height:
        scale = target_height / float(current_height)
        target_dim = (int(current_width * scale), target_height)
    else:
        scale = target_width / float(current_width)
        target_dim = (target_width, int(current_height * scale))

    return cv2.resize(image, target_dim, interpolation=inter)


def random_file(path):
    """Randomly selects a file from a path"""
    return choice(glob(path))


def model_exists(client, project_name, model_version):
    """Checks if a Lookout for Vision model exists"""
    try:
        client.describe_model(
            ProjectName=project_name,
            ModelVersion=model_version
        )
        return True
    except client.exceptions.from_code('ResourceNotFoundException'):
        return False


def list_projects(client):
    """Returns a list of Lookout for Vision projects"""
    response = client.list_projects()
    return [p['ProjectName'] for p in response['Projects']]


def list_models(client, project_name):
    """Returns available model versions for a given Lookout for Vision project"""
    try:
        response = client.list_models(
            ProjectName=project_name
        )
        return [m['ModelVersion'] for m in response['Models']]
    except client.exceptions.from_code('ResourceNotFoundException') as error:
        raise error


def describe_project(client, project_name):
    """Returns information about a Lookout for Vision Project"""
    response = client.describe_project(
        ProjectName=project_name
    )
    return response['ProjectDescription']


def describe_model(client, project_name, model_version):
    """Returns information about a Lookout for Vision model"""
    response = client.describe_model(
        ProjectName=project_name,
        ModelVersion=model_version
    )
    return response['ModelDescription']


# pylint: disable=R0913
def add_text(image, text, coords,
             font=cv2.FONT_HERSHEY_PLAIN,
             font_scale=1,
             color=(255, 255, 255),
             thickness=1):
    """Adds text centered around a set of coordinates to an image"""
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = coords[0] - text_size[0] // 2
    text_y = coords[1] - text_size[1] // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)


def run_single_detection(client, project_name, model_version,
                         img, display=True, wait=DEFAULT_WAITING_PERIOD):
    """Run anomaly detection on a single image"""

    # Run anomaly detection
    with open(img, 'rb') as img_f:
        response = client.detect_anomalies(
            ProjectName=project_name,
            ModelVersion=model_version,
            Body=bytearray(img_f.read()),
            ContentType=set_content_type(img)
        )

    # Get a prediction
    detection_results = response['DetectAnomalyResult']
    if detection_results['IsAnomalous']:
        prediction = "ANOMALY"
    else:
        prediction = "NORMAL"

    # Get confidence value
    confidence = detection_results['Confidence']

    # Display detection results on top of the image
    if display:
        image = resize_with_aspect_ratio(cv2.imread(img), target_height=400)
        text = "Prediction: %s | Confidence: %.2f" % (prediction, confidence * 100)
        add_text(image, text, (image.shape[1] // 2, image.shape[0] - DEFAULT_TEXT_PADDING))

    # Check if the prediction is correct and label the image
    if prediction.lower() in img:
        logging.info("%s | Prediction: %s | Confidence: %.2f%% --> %s",
                        img, prediction, confidence * 100, CORRECT_PREDICTION_LABEL)
        if display:
            add_text(image,
                        CORRECT_PREDICTION_LABEL,
                        (image.shape[1] // 2, DEFAULT_TEXT_PADDING),
                        font_scale=2,
                        color=(0, 255, 0),
                        thickness=2)
    else:
        logging.error("%s | Prediction: %s | Confidence: %.2f%% --> %s",
                        img, prediction, confidence * 100, INCORRECT_PREDICTION_LABEL)
        if display:
            add_text(image,
                        INCORRECT_PREDICTION_LABEL,
                        (image.shape[1] // 2, DEFAULT_TEXT_PADDING),
                        font_scale=2,
                        color=(0, 0, 255),
                        thickness=2)

    # Display image + results
    if display:
        cv2.imshow(DEFAULT_WINDOW_NAME, image)
        cv2.waitKey(wait*1000)

    prediction_results = {
        "image": img,
        "prediction": prediction,
        "confidence": confidence
    }

    return prediction_results


def write_to_csv(results, filename='results.csv'):
    """Write results to a CSV file"""
    DataFrame(results).to_csv(filename)


def run_multiple_detections(client, project_name, model_version,
                            image_folder, display=True, wait=DEFAULT_WAITING_PERIOD,
                            run='sequential', save='results.csv'):
    """Run anomaly detection on multiple images"""

    # Open cv2 window
    if display:
        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Check if model exists
    assert model_exists(client, project_name, model_version), "Model does not exist"

    results = []
    try:
        # Run anomaly detection on all images exactly once
        if run == 'sequential':
            for img in glob('{}/*'.format(image_folder)):
                result = run_single_detection(client,
                                              project_name,
                                              model_version,
                                              img,
                                              display,
                                              wait)
                results.append(result)
        # Pick images at random and run anomaly detection
        elif run == 'random':
            while True:
                img = random_file('{}/*'.format(image_folder))
                result = run_single_detection(client,
                                              project_name,
                                              model_version,
                                              img,
                                              display,
                                              wait)
                results.append(result)
        else:
            logging.error("Invalid run method: %s", run)
    except KeyboardInterrupt:
        pass

    # Save results to a CSV file
    if save:
        logging.info("Saving results")
        write_to_csv(results, save)

def parse_args(args):
    """Parse command line parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p',
                        type=str,
                        required=True,
                        help='the project name')
    parser.add_argument('--model', '-m',
                        type=str,
                        default='1',
                        help='the model version')
    parser.add_argument('--images', '-i',
                        type=str,
                        default='extra_images',
                        help='either a file or a folder containing images')
    parser.add_argument('--wait', '-w',
                        type=int,
                        default=5,
                        help='waiting period before loading the next image')
    parser.add_argument('--run', '-r',
                        choices=['sequential', 'random'],
                        default='sequential',
                        help='method for image selection')
    parser.add_argument('--display', '-d',
                        action='store_true',
                        help='Whether to display the image and prediction results')
    parser.add_argument('--save', '-s',
                        type=str,
                        help='Save the results to a CSV file')
    return parser.parse_args(args)


def splash_screen():
    """Print splash screen"""
    print(__doc__)


def destroy_all_windows():
    """Destroys all cv2 windows
    For more information, please check:
    https://medium.com/@mrdatainsight/how-to-use-opencv-imshow-in-a-jupyter-notebook-quick-tip-ce83fa32b5ad"""
    cv2.destroyAllWindows()


def main(client, args):
    """Main function"""

    # Print splash screen
    splash_screen()

    # Parse arguments
    args = parse_args(args)

    # Check project
    logging.info("Validating project")
    assert args.project in list_projects(client),\
            "Project '{}' does not exist".format(args.project)
    logging.debug("Project: %s", describe_project(client, args.project))

    # Check model version
    logging.info("Validating model")
    assert args.model in list_models(client, args.project),\
            "Model '{}' does not exist".format(args.model)
    logging.debug("Model: %s", describe_model(client, args.project, args.model))

    # Check if the specified folder exists
    logging.info("Checking images: %s", args.images)
    assert os.path.isdir(args.images) or os.path.isfile(args.images),\
            "'{}' should be a folder or an image file".format(args.images)

    # Run anomaly detection
    logging.info("Starting anomaly detection")
    if os.path.isfile(args.images):
        run_single_detection(client,
                             args.project,
                             args.model,
                             args.images,
                             display=args.display,
                             wait=args.wait,)
    elif os.path.isdir(args.images):
        run_multiple_detections(client,
                                args.project,
                                args.model,
                                args.images,
                                display=args.display,
                                wait=args.wait,
                                run=args.run,
                                save=args.save)

    # Destroy all windows and exit application
    if args.display:
        logging.info("Destroying all windows")
        destroy_all_windows()

    logging.info("Exiting application")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # Initialize Lookout for Vision client
    lookout_vision = boto3.client('lookoutvision')

    # Run anomaly detection
    main(lookout_vision, sys.argv[1:])
