from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd 
import os 

# Insert the path to a file which contains test predictions
PRED_FILE = "./ensembles/14_xception_da_ftall_1e4.csv"

# Read the predictions
pred_df = pd.read_csv(PRED_FILE)

# Insert the path to the test folder to read the images
TEST_DIR = "./Test 2/Test 2"

# Get test labels from the test images
test_labels = [int(file[:file.index("_")]) for file in os.listdir(TEST_DIR) if file != ".DS_Store"]
test_dict = dict(zip(os.listdir(TEST_DIR), test_labels))

# Plot the confusion matrix. The labels should be numbers
conf_matrix = confusion_matrix(list(test_dict.values()), list(map(lambda x: int(x), list(pred_df["label"].values))))
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot(include_values=False, cmap=plt.cm.coolwarm)
plt.show()