# Background Remover with Google Vertex AI

This Streamlit application uses Google Cloud's Vertex AI to remove backgrounds from images. It provides a simple, user-friendly interface for uploading images, processing them, and downloading the results.

## Features

- Upload JPG, JPEG, or PNG images
- Choose between removing background or foreground
- Real-time image processing using Google's Vertex AI
- Download processed images

## Setup Instructions

### Prerequisites

1. A Google Cloud account with Vertex AI API enabled
2. Python 3.8 or later installed
3. Appropriate IAM permissions for Vertex AI

### Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up Google Cloud authentication:

```bash
# Install gcloud CLI if you haven't already
# https://cloud.google.com/sdk/docs/install

# Authenticate with Google Cloud
gcloud auth application-default login
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

### Using the App

1. Enter your Google Cloud Project ID in the sidebar
2. Select your preferred location and mask mode
3. Upload an image
4. Click the "Remove background" or "Remove foreground" button (depending on your selection)
5. Wait for processing to complete
6. Download the processed image

## Important Notes

- Processing large images may take some time
- Ensure you have billing set up on your Google Cloud project
- The app uses the Vertex AI ImageGeneration model, which may incur costs according to Google Cloud's pricing

## Troubleshooting

- If you encounter authentication errors, ensure you've run `gcloud auth application-default login`
- Verify that the Vertex AI API is enabled in your Google Cloud project
- Check that your Google Cloud project has billing enabled