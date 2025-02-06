import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight but effective model

# Read the updated datasets
breed_traits = pd.read_csv(r"/Users/anapaula.martinelli/Desktop/petfinderdataset/breed_traits_long_final1.csv")
trait_descriptions = pd.read_csv(r"/Users/anapaula.martinelli/Desktop/petfinderdataset/trait_descriptioncopiaw.csv")

# Remove specific breed if necessary
breed_traits['Trait_Score'] = pd.to_numeric(breed_traits['Trait_Score'], errors='coerce')
numeric_traits = breed_traits[pd.notna(breed_traits['Trait_Score'])]
breed_traits_wide = numeric_traits.pivot(index='Breed', columns='Trait', values='Trait_Score')

# Define size categories explicitly
size_mapping = {
    "Small Dog": "small",
    "Medium Dog": "medium",
    "Big Dog": "large"
}

# Ensure Size_Category is populated correctly
if 'Size' in breed_traits_wide.columns:
    breed_traits_wide['Size_Category'] = breed_traits_wide['Size'].map(size_mapping)
    print("Unique size categories in dataset:", breed_traits_wide['Size_Category'].unique())  # Debugging print

# Size keywords for preference analysis with numerical mappings
size_keywords = {
    1: ["very small dog", "tiny dog", "miniature dog", "smallest dog", "toy dog", "teacup dog"],
    2: ["small dog", "little dog", "compact dog", "smaller dog"],
    3: ["medium dog", "medium-sized dog", "moderate sized dog", "average sized dog", "regular sized dog"],
    4: ["large dog", "big dog", "bigger dog", "tall dog"],
    5: ["very large dog", "giant dog", "huge dog", "biggest dog", "massive dog", "extra large dog"]
}

def analyze_description(description):
    description = description.lower()  # Convert once for all checks
    size_preference = "any"
    coat_preference = "any"
    
    # Check direct word matches FIRST
    # Large/Big (4)
    if any(word in description for word in ["large", "big"]):
        print("\nSize preference detected through direct match: 4 (Large)")
        size_preference = 4
    # Very Large (5)
    elif any(word in description for word in ["huge", "giant", "very large", "massive"]):
        print("\nSize preference detected through direct match: 5 (Very Large)")
        size_preference = 5
    # Very Small (1)
    elif any(word in description for word in ["tiny", "very small", "miniature", "toy"]):
        print("\nSize preference detected through direct match: 1 (Very Small)")
        size_preference = 1
    # Small (2)
    elif "small" in description:
        print("\nSize preference detected through direct match: 2 (Small)")
        size_preference = 2
    # Medium (3)
    elif any(word in description for word in ["medium", "average", "moderate"]):
        print("\nSize preference detected through direct match: 3 (Medium)")
        size_preference = 3
    
    # Check coat length preference
    if "short" in description or "smooth" in description:
        print("\nCoat preference detected: short")
        coat_preference = "short"
    elif "long" in description or "fluffy" in description or "shaggy" in description:
        print("\nCoat preference detected: long")
        coat_preference = "long"
    elif "medium" in description and ("coat" in description or "hair" in description):
        print("\nCoat preference detected: medium")
        coat_preference = "medium"
    
    print(f"\nFinal preferences - Size: {size_preference}, Coat: {coat_preference}")
    return (size_preference, coat_preference)  # Explicitly return both preferences as a tuple

def score_dog(preferences, breed_data):
    size_pref, coat_pref = preferences
    scored_breeds = []
    
    print(f"\nDebug - Starting scoring with preferences:")
    print(f"Size preference: {size_pref}")
    print(f"Coat preference: {coat_pref}")
    
    # Define strict size ranges for each preference
    size_ranges = {
        1: [1],           # Very Small: only size 1
        2: [1, 2],        # Small: sizes 1-2
        3: [3],           # Medium: only size 3
        4: [4],           # Large: only size 4
        5: [4, 5]         # Very Large: sizes 4-5
    }
    
    # Convert breed_data to wide format if it's not already
    if 'Trait' in breed_data.columns:
        breed_data_wide = breed_data.pivot(index='Breed', columns='Trait', values='Trait_Score')
    else:
        breed_data_wide = breed_data
    
    for breed, data in breed_data_wide.iterrows():
        score = 0
        explanations = []
        include_breed = True
        
        # Debug print for each breed
        print(f"\nChecking {breed}:")
        print(f"Size: {data.get('Size')}")
        print(f"Coat Length: {data.get('Coat Length')}")
        
        # Size preference scoring
        if size_pref != "any":
            breed_size = data.get('Size')
            if breed_size not in size_ranges[size_pref]:
                print(f"Size mismatch - wanted {size_ranges[size_pref]}, got {breed_size}")
                include_breed = False
            else:
                score += 50
                explanations.append("Matches desired size category")
                print("Size matches!")
        
        # Coat length scoring
        if coat_pref != "any" and include_breed:
            breed_coat = data.get('Coat Length')
            if pd.notna(breed_coat):
                if coat_pref == "short" and breed_coat <= 2:
                    score += 30
                    explanations.append("Matches desired short coat")
                    print("Coat matches!")
                elif coat_pref == "medium" and breed_coat == 3:
                    score += 30
                    explanations.append("Matches desired medium coat")
                    print("Coat matches!")
                elif coat_pref == "long" and breed_coat >= 4:
                    score += 30
                    explanations.append("Matches desired long coat")
                    print("Coat matches!")
                else:
                    include_breed = False
                    print(f"Coat mismatch - wanted {coat_pref}, got length {breed_coat}")
        
        # Only add breeds that match both size and coat preferences
        if include_breed:
            # Additional trait scoring
            for trait, value in data.items():
                if trait not in ['Size', 'Coat Length'] and pd.notna(value):
                    score += value
                    explanations.append(f"Good match for {trait}")
            
            scored_breeds.append((breed, score, explanations))
            print(f"Added to matches with score {score}")
    
    print(f"\nTotal matches found: {len(scored_breeds)}")
    return scored_breeds

def main():
    print("Welcome to the Dog Matching System!\n")
    user_description = input("Please describe your ideal dog (including size and coat preferences): ")
    
    # Analyze description for preferences - get both size and coat preferences
    preferences = analyze_description(user_description)  # This returns a tuple (size, coat)
    
    # Score matching dogs with both preferences
    scored_dogs = score_dog(preferences, breed_traits_wide)  # Pass the entire preferences tuple
    
    if not scored_dogs:
        print("\nNo matching dogs found based on your preferences.")
    else:
        print("\nHere are some dog breeds that match your preferences:")
        for breed, score, explanations in scored_dogs[:5]:  # Show top 5 matches
            print(f"\n{breed}:")
            print(f"Total Score: {score}")
            for exp in explanations:
                print(f"  - {exp}")

if __name__ == "__main__":
    main()
