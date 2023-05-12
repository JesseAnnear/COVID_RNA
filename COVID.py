import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

TEST_PATH = 'Asn2DataSet\\TestingDataset'
ALPHA_PATH = 'Asn2DataSet\\TrainingDataset\\Alpha'
BETA_PATH = 'Asn2DataSet\\TrainingDataset\\Beta'
DELTA_PATH = 'Asn2DataSet\\TrainingDataset\\Delta'
GAMMA_PATH = 'Asn2DataSet\\TrainingDataset\\Gamma'


def read_fasta_file(filename):
    """reads the fasta files"""
    sequences = {}
    with open(filename, "r") as file:
        sequence_id = None
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id is not None:
                    sequences[sequence_id] = sequence
                sequence_id = line[1:]
                sequence = ""
            else:
                sequence += line
        if sequence_id is not None:
            sequences[sequence_id] = sequence
    return sequences


def flatten_matrix(matrix):
    """Function to flatten the matrix """
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    flattened = [0] * (num_rows * num_cols)
    for j in range(num_cols):
        for i in range(num_rows):
            flattened[i * num_cols + j] = matrix[i][j]
    return flattened


# Note for professor
# I had to remake this function as the one provided was giving me a matrix of all zero's for some reason
def cgr(seq, order, k):
    """Translates strings into integers using cgr function"""
    ln = len(seq)
    pw = 2 ** k
    out = [[0 for i in range(pw)] for j in range(pw)]
    x = 0
    y = 0
    for i in range(k - 1, ln):
        x = (x << 1) & (pw - 1)
        y = (y << 1) & (pw - 1)
        if seq[i] == order[2] or seq[i] == order[3]:
            x |= 1
        if seq[i] == order[0] or seq[i] == order[3]:
            y |= 1
        out[y][x] += 1
    return out


# Create a number of dictions for all types of covid and the test set
# The first set denotes the list of filenames per directory
ALPHA_DIR = os.listdir(ALPHA_PATH)
BETA_DIR = os.listdir(BETA_PATH)
DELTA_DIR = os.listdir(DELTA_PATH)
GAMMA_DIR = os.listdir(GAMMA_PATH)
TEST_DIR = os.listdir(TEST_PATH)

# These directories are used to store values for each type
ALPHA_DICT = {}
BETA_DICT = {}
DELTA_DICT = {}
GAMMA_DICT = {}
TEST_DICT = {}

# These dictionaries hold the CGR values for each subtype
ALPHA_CGR = {}
BETA_CGR = {}
DELTA_CGR = {}
GAMMA_CGR = {}
TEST_CGR = {}

# ALPHA: loops through the files and assigns it to a key
for i in ALPHA_DIR:
    tester = read_fasta_file(ALPHA_PATH + '/' + i)
    key = list(tester.keys())
    ALPHA_DICT[i] = tester.get(key[0])

# ALPHA: loops through the key to both get cgr representation and flatten the matrix
for key, value in ALPHA_DICT.items():
    cgrs = cgr(value, order='ACGT', k=7)
    ALPHA_CGR[key] = flatten_matrix(cgrs)

# BETA: loops through the files and assigns it to a key
for i in BETA_DIR:
    tester = read_fasta_file(BETA_PATH + '/' + i)
    key = list(tester.keys())
    BETA_DICT[i] = tester.get(key[0])

# BETA: loops through the key to both get cgr representation and flatten the matrix
for key, value in BETA_DICT.items():
    cgrs = cgr(value, order='ACGT', k=7)
    BETA_CGR[key] = flatten_matrix(cgrs)

# DELTA: loops through the files and assigns it to a key
for i in DELTA_DIR:
    tester = read_fasta_file(DELTA_PATH + '/' + i)
    key = list(tester.keys())
    DELTA_DICT[i] = tester.get(key[0])

# DELTA: loops through the key to both get cgr representation and flatten the matrix
for key, value in DELTA_DICT.items():
    cgrs = cgr(value, order='ACGT', k=7)
    DELTA_CGR[key] = flatten_matrix(cgrs)

# GAMMA: loops through the files and assigns it to a key
for i in GAMMA_DIR:
    tester = read_fasta_file(GAMMA_PATH + '/' + i)
    key = list(tester.keys())
    GAMMA_DICT[i] = tester.get(key[0])

# GAMMA: loops through the key to both get cgr representation and flatten the matrix
for key, value in GAMMA_DICT.items():
    cgrs = cgr(value, order='ACGT', k=7)
    GAMMA_CGR[key] = flatten_matrix(cgrs)

# unpack the 4 types into a single data dictionary and get the labels
data = {**ALPHA_CGR, **BETA_CGR, **DELTA_CGR, **GAMMA_CGR}
labels = ['ALPHA'] * len(ALPHA_CGR) + ['BETA'] * len(BETA_CGR) + ['DELTA'] * len(DELTA_CGR) + ['GAMMA'] * len(GAMMA_CGR)

# Turn both values and labels into numpy array
data_array = np.array(list(data.values()))
labels_array = np.array(labels)

## 2. The number that should divide these vectors by is the maxium value of each vector.
## I used a MinMaxscaler from sklearn in order to achieve this.
scaler = MinMaxScaler()
data_array = scaler.fit_transform(data_array)

# Set up for k-folds
kf = KFold(n_splits=10, shuffle=True)

# loop through the k-fold splits
for train_index, test_index in kf.split(data_array):
    # create the train and test sets
    X_train, X_test = data_array[train_index], data_array[test_index]
    y_train, y_test = labels_array[train_index], labels_array[test_index]

    # Train the data on the support vector machine
    # decided to start with linear and see how it goes
    # my accuracy was 1.0 therefore it seems the data is in fact linearly separable
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Now we predict or values and calculate the score on our test values
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)


# Now we classify the test set of fasta files

# loops through the files and assigns it to a key
for i in TEST_DIR:
    tester = read_fasta_file(TEST_PATH + '/' + i)
    key = list(tester.keys())
    TEST_DICT[i] = tester.get(key[0])

# loops through the key to both get cgr representation and flatten the matrix
for key, value in TEST_DICT.items():
    cgrs = cgr(value, order='ACGT', k=7)
    TEST_CGR[key] = flatten_matrix(cgrs)

# Turn test set into a cgr array
TEST_ARRAY = np.array(list(TEST_CGR.values()))

# Scale to a range between zero and one like before
TEST_ARRAY = scaler.fit_transform(TEST_ARRAY)

# We predict the fasta files from the test set
predicted_labels = svm.predict(TEST_ARRAY)
