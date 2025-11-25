## **Random Forest Model for Spike Classification**
This module allows the user to annotate fluorescence trace data and train a random forest classifier on it. There are three components. 
1. **prepare_data.py**
This script aggregates data from a specified ROI dictionary and parses it to create features using imports from the utils module. These features are saved to the same dictionary from which the original data was pulled. 
2. **annotate_spikes**
This script creates a GUI that loads individual spikes from the now complete ROI dictionary. The GUI visualizes a raw and smooth fluorescence trace and highlights the region describing the spike. The "good", "bad", and "skip" buttons control the labeling or lackthereof for a displayed spike. These labels are saved to the same dictionary. 
3. **train_classifier**
This script trains a random forest model on the annotated spike features. Random forest is adept at handling the nonlinear relationship between spike features and the spike's label. Originally engineering 18 features, this script now isolates the 3 most predictive and influential features for the dataset: 
- Spike prominence: the raw value describing the extent to which a spike rises from its theoretical baseline. 
- Dominance Score: the ratio of a spike's prominence to that of its parent peak, identified by a clustering algorithm dependent on the median spike width. 
