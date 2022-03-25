# Required libraries
import os 
import pandas as pd

# Insert the path to the test folder to read the images
TEST_DIR = "./Test 2/Test 2"
# List of prediction files
PRED_FILES = [
    "./ensembles/14_xception_da_ftall_1e4.csv",
    "./ensembles/17_xception_da_ftall_1e4_35e.csv",
    "./ensembles/23_inception_ftall_1e5.csv",
]

# Load the list of predictions from the different classifiers 
# in dictionaries whose keys are the test filenames and whose values
# are the test labels
pred_list = []
for pred_file in PRED_FILES:
    pred_df = pd.read_csv(pred_file)
    pred_list.append(dict(zip(list(pred_df["id.jpg"].values), list(pred_df["label"].values))))

# Get test labels from the test images
test_labels = [file[:file.index("_")] for file in os.listdir(TEST_DIR)]
test_dict = dict(zip(os.listdir(TEST_DIR), test_labels))
# Delete weird key
del test_dict[".DS_Store"]

# Iterate over the test images to choose the most voted class between the classifiers
ensemble_preds = []
# Calculate the accuracy over the test dataset
accuracy = 0
for test_img in test_dict:
    try:
        all_preds = [pred_dict[test_img] for pred_dict in pred_list]
        most_voted_class = max(set(all_preds), key = all_preds.count)
        ensemble_preds.append(most_voted_class)
        accuracy += 1 if int(test_dict[test_img]) == int(most_voted_class) else 0
    except:
        pass
    
print("Accuracy: ", (accuracy/len(test_dict))*100)

# Create submission file
submission_df = pd.DataFrame({'id.jpg': list(test_dict.keys()), 'label': ensemble_preds})
submission_df.to_csv("27_ensemble_submission.csv", index=False)