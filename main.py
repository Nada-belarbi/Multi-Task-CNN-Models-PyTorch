import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your four model classes from your models.py file
try:
    from models import ImageClassifier, ImageSegmenter, BBoxRegressor, ImageGenerator
    print("Successfully imported models.")
except ImportError:
    print("Could not import models from models.py. Make sure the file exists and classes are named correctly.")
    exit()

if __name__ == "__main__":

    # Define common input parameters
    BATCH_SIZE = 4
    IMG_HEIGHT = 512
    IMG_WIDTH = 512

    # Create a dummy input tensor
    dummy_input = torch.rand(BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH)

    print(f"--- Testing Models with Input Shape: {list(dummy_input.shape)} ---\n")

    # --- Test 1: Image Classifier ---
    print("=" * 30)
    print("Task 1: Image Classification")
    print("=" * 30)
    try:
        classifier_model = ImageClassifier(num_classes=10)
        output = classifier_model(dummy_input)

        print(f"Input shape: {list(dummy_input.shape)}")
        print(f"Output shape: {list(output.shape)}")
        print("Expected shape: [B, 10] -> [4, 10]")
        assert list(output.shape) == [BATCH_SIZE, 10]
        print("Test PASSED!\n")

        # --- Bonus Test for Adaptive Pooling ---
        print("Bonus Test: Testing with 256x256 input ...")
        dummy_input_small = torch.rand(BATCH_SIZE, 3, 256, 256)
        output_small = classifier_model(dummy_input_small)
        assert list(output_small.shape) == [BATCH_SIZE, 10]
        print("Bonus Test PASSED! Model is flexible.\n")

    except Exception as e:
        print(f"Test FAILED: {e}\n")

    # --- Test 2: Image Segmenter ---
    print("=" * 30)
    print("Task 2: Image Segmentation")
    print("=" * 30)
    try:
        segmenter_model = ImageSegmenter(num_classes=5)
        output = segmenter_model(dummy_input)

        print(f"Input shape: {list(dummy_input.shape)}")
        print(f"Output shape: {list(output.shape)}")
        print("Expected shape: [B, num_classes, H, W] -> [4, 5, 512, 512]")
        assert list(output.shape) == [BATCH_SIZE, 5, IMG_HEIGHT, IMG_WIDTH]
        print("Test PASSED!\n")

    except Exception as e:
        print(f"Test FAILED: {e}\n")

    # --- Test 3: Bounding Box Regressor ---
    print("=" * 30)
    print("Task 3: Bounding Box Regression")
    print("=" * 30)
    try:
        regressor_model = BBoxRegressor(num_coords=4)
        output = regressor_model(dummy_input)

        print(f"Input shape: {list(dummy_input.shape)}")
        print(f"Output shape: {list(output.shape)}")
        print("Expected shape: [B, 4] -> [4, 4]")
        assert list(output.shape) == [BATCH_SIZE, 4]
        print("Test PASSED!\n")

        # --- Bonus Test for Adaptive Pooling ---
        print("Bonus Test: Testing with 256x256 input ...")
        dummy_input_small = torch.rand(BATCH_SIZE, 3, 256, 256)
        output_small = regressor_model(dummy_input_small)
        assert list(output_small.shape) == [BATCH_SIZE, 4]
        print("Bonus Test PASSED! Model is flexible.\n")

    except Exception as e:
        print(f"Test FAILED: {e}\n")

    # --- Test 4: Image Generator ---
    print("=" * 30)
    print("Task 4: Image Generation")
    print("=" * 30)
    try:
        generator_model = ImageGenerator(in_channels=3, out_channels=3)
        output = generator_model(dummy_input)

        print(f"Input shape: {list(dummy_input.shape)}")
        print(f"Output shape: {list(output.shape)}")
        print("Expected shape: [B, 3, H, W] -> [4, 3, 512, 512]")
        assert list(output.shape) == [BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH]
        print("Test PASSED!\n")

    except Exception as e:
        print(f"Test FAILED: {e}\n")
