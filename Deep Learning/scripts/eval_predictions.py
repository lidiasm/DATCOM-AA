# Required libraries
import os 
import pandas as pd

# Insert the path to the test folder to read the images
TEST_DIR = "../../../Prácticas/Deep Learning/Test 2/Test 2"
# Insert the path and filename of the predictions
PRED_FILE = "../../../Prácticas/Deep Learning/Predictions/1_xception_test.csv"

# Get test labels from the test images
test_labels = [file[:file.index("_")] for file in os.listdir(TEST_DIR)]
test_dict = dict(zip(os.listdir(TEST_DIR), test_labels))

# Read the test predictions
pred_labels = pd.read_csv(PRED_FILE)
pred_dict = dict(zip(list(pred_labels["id.jpg"]), list(pred_labels["label"])))

# Compute the accuracy
accuracy = 0
for test_image in test_dict:
    try:
        accuracy += 1 if int(test_dict[test_image]) == int(pred_dict[test_image]) else 0
    except:
        pass

print("Accuracy: ", (accuracy/len(test_labels))*100)
