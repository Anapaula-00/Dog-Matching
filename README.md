
# Dog Matching - Recommendation System for Choosing the Right Dog
## Project description
The **Dog Matching** project aims to help individuals select the most suitable dog breed based on their preferences and lifestyle. Using advanced **Natural Language Processing (NLP)** techniques, the system analyzes the characteristic traits of various dog breeds and provides recommendations based on personalized criteria.

By leveraging the **Sentence Transformers** model, the software compares textual descriptions and numerical scores to generate recommendations based on semantic similarity and objective values.

The system can be useful for:

- Future dog owners looking for a breed that suits their needs.
- Shelters and animal welfare organizations to assist in selecting the best match between dogs and adopters.
- Veterinarians and trainers to provide recommendations based on data and scientific characteristics.


## Technologies Used
- **Python 3** - Main programming language
- **Pandas & NumPy** - For data processing and manipulation
- **Sentence Transformers** - For semantic similarity calculations
- **TensorFlow** *(optional)* - For NLP model optimization
- **Logging** - For error handling and event logging

## Installation
### Prerequisites
Ensure you have Python 3 installed on your system. If not, download and install it from [python.org](https://www.python.org/).

To install all required dependencies, run the following command:

```sh
pip install pandas numpy sentence-transformers tensorflow
```

If you do not intend to use TensorFlow, you can omit it:
```sh
pip install pandas numpy sentence-transformers
```

### Downloading the Datasets
The project relies on two essential CSV files containing information on dog breeds:
1. **`breed_traits.csv`** - Contains dog breeds and their scores for various traits.
2. **`trait_descriptions.csv`** - Contains detailed descriptions of the traits.

Make sure to download these files and place them in the correct directory.

## How to Use the Program
### Main Steps:
1. **Loading the NLP Model**: The system uses `all-MiniLM-L6-v2`, a lightweight yet effective model for sentence comparison.
2. **Reading the Datasets**: The CSV files are read and transformed into an analyzable format.
3. **Data Preprocessing**:
   - Converting trait scores into numerical format
   - Removing breeds with incomplete data
   - Creating a `wide format` structure for better data management
4. **Defining Size Categories**:
   - Small (**small**) - Examples: Chihuahua, Maltese
   - Medium (**medium**) - Examples: Border Collie, Bulldog
   - Large (**large**) - Examples: Great Dane, Labrador Retriever
5. **Processing the Matching**:
   - Calculating similarity between trait characteristics
   - Generating personalized recommendations

### Running the Program
To run the script, use the following command:
```sh
python pet_matching_code.py
```

The program will analyze the data and provide a list of the most suitable dog breeds based on the given criteria.

## Data Structure
### `breed_traits.csv` Dataset
| Column         | Description                                      |
|---------------|------------------------------------------------|
| Breed        | Name of the dog breed                          |
| Trait        | Name of the dog's characteristic trait         |
| Trait_Score  | Numerical rating of the trait                 |
 |

### `trait_descriptions.csv` Dataset
| Column         | Description                                    |
|---------------|----------------------------------------------|
| Trait        | Name of the characteristic trait             |
| Description  | Detailed explanation of the trait            |


## Logging and Debugging
The code includes a logging system to track issues and analyze execution:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Execution started...")
```
To enable debug mode, change the logging level to `DEBUG`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Author
This project was developed as part of a university exam for the Data Mining and Text Analytics course

