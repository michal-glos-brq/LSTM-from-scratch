                                    #####################################
                                    #@$%&                           &%$@#
                                    #!      SFC - Projekt LSTM         !#
                                    #!          Michal Glos            !#
                                    #!           xglosm01              !#
                                    #!              __                 !#
                                    #!            <(o )___             !#
                                    #!             ( ._> /             !#
                                    #!              `---'              !#
                                    #@$%&                           &%$@#
                                    #####################################

# This humble python module is responsible for handling (loading and processing)
# amazon reviews (of video games) text data to be fed into LSTM neural network

import gzip
import json
from tqdm import tqdm
import numpy as np
import time
import os
import re
import pickle
import requests

# Constatnts
EMBED = 'data/embed.p'
DATA_PATH = 'data/Video_Games.json'
DOWNLOAD_PATH = "data/Video_Games.json.gz"
DATA_URL = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Video_Games.json.gz'

# Regular expression to choose just alphanumerical chars from a word
regex = re.compile('[\W_]+')


def assert_files_safe_dump(path):
    '''
    Create neccessary directories to safely create file.

    Input arguments:
        path:   Path to the file to be created
    '''
    # get the file parent dir
    parent_dir = '/'.join(path.split('/')[:-1:])
    # Create the parent dir - if does not already exist
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def _download_data(data_path=DATA_PATH, download_path=DOWNLOAD_PATH, data_url=DATA_URL):
    '''
    Download and/or extract data from amazon servers.

    Optional parameters:
        data_path:      Path to a file, where downloaded data would be extracted
        download_path:  Path, where the file would be downloaded
        data_url:       URL to data
    
    Returns:
        bool:           Was download succesful?
    '''
    # If downloaded gzip does not exist, fetch data from data_url
    if not os.path.exists(download_path):
        print(f"Will now download data from: {data_url}")
        # Fake the headers
        headers = {
                'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'
                'AppleWebKit/537.36 (KHTML, like Gecko)'
                'Chrome/39.0.2171.95 Safari/537.36')
        }
        # Request the file
        data_request = requests.get(data_url, headers=headers, stream=True)
        total_size = int(data_request.headers.get('content-length', 0))
        # Create progress bar to show download progress
        tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True, ncols=100)
        # If the request was succesfull, collect downloaded data from strem and dump it into a file
        if data_request.status_code == 200:
            assert_files_safe_dump(download_path)
            with open(download_path, 'wb') as f:
                for chunk in data_request.iter_content(chunk_size=1024):
                    tqdm_bar.update(len(chunk))
                    f.write(chunk)
            # TQDM 'hack' to stop printing the progress bar before the print statement
            time.sleep(0.5)
            print("\nData were succesfully downloaded")
        else:
            # Request failed:
            print("Download failed, please try it once again or download data" + \
            "manually from: " + data_url + " and save the json file here.")
            return False

    # Extract gzip here
    with gzip.open(download_path, 'rb') as f:
        assert_files_safe_dump(data_path)
        # Merlin could not work with files larger then +1GB split it ... :(
        data = [line.decode('UTF-8') for line in f.readlines()]
        data1 = data[:int(len(data)/2)]
        data2 = data[int(len(data)/2):]
        with open(data_path, 'w') as data_file:
            data_file.write(''.join(data1))
        with open(data_path.replace(".json", "1.json"), 'w') as data_file:
            data_file.write(''.join(data2))
        return True

def _load_data(data_path=DATA_PATH):
    '''
    Load Amazon review data from fixed path stored in module.
    
    Optional parameters:
        data_path:  Path to a file with requested data
    
    Returns:
        list:   list of dictionaries representing single data entries
    '''
    # If data does not exist, download it
    if not ( os.path.exists(data_path) and os.path.exists(data_path.replace(".json", "1.json")) ):
        data_downloaded = _download_data(data_path=data_path)
        # If download failed, exit
        if not data_downloaded:
            exit(69)

    # Load the data
    data = []
    with open(data_path, 'r') as f:
        for line in tqdm(f.readlines(), desc="Loading data (part 1)", ncols=100):
            data.append(json.loads(line))
    with open(data_path.replace(".json", "1.json"), 'r') as f:
        for line in tqdm(f.readlines(), desc="Loading data (part 2)", ncols=100):
            data.append(json.loads(line))
    # TQDM 'hack' to stop printing the progress bar before the print statement
    time.sleep(0.5)
    print("Loaded " + str(len(data)) + " entries of data")
    return data


def _filter_json_entries(data, keys=[]):
    '''
    From list of dicts, return list of subdicts with input keys.
    
    Input arguments:
        data:   Data to be filtered - list of dictionaries
        keys:   List of keys to dictionaries in data
    
    Returns:
        list of dictionaries:   Dictionaries will contain only data
                                accessible with provided keys
    '''
    filtered_data = []
    # Iterate through data and pickl only parts accessible with keys
    for entry in tqdm(data, desc="Processing data", ncols=100):
        ### If all requested keys are present, pick them
        if all([ key in entry.keys() for key in keys ]):
            filtered_entry = {key: entry[key] for key in keys}
            filtered_data.append(filtered_entry.copy())

    # TQDM 'hack' to stop printing the progress bar before the print statement
    time.sleep(0.5)
    print(str(len(filtered_data)) + " succesfully processed")
    print(str(len(data) - len(filtered_data)) + " filtered out")
    return filtered_data


def _filter_short_data(data, min_len=64, data_key='reviewText'):
    '''
    Filter out short data.

    Input arguments:
        data:   Data to be filtered - list of dictionaries
    
    Optional arguments:
        min_len:    Minimal requested length of data
        data_key:   Key to measured data
    
    Returns:
        list of dictionaries:   Dictionaries only with data accessed wit data_key
                                longer then min_len
    '''
    filtered = []
    short = 0
    # Iterate through data and select only data longer then min_len
    for d in tqdm(data, desc="Filtering short data", ncols=100):
        if len(d[data_key]) > min_len:
            filtered.append(d)
        else:
            # Count the short entries
            short += 1
    # TQDM 'hack' to stop printing the progress bar before the print statement
    time.sleep(0.5)
    print(f"Good data: {len(filtered)}\nBad data: {short}")
    return filtered
 

def _eliminate_apriori_inequity(data, key, one_class_size=None):
    '''
    Get the same amount of data for every value accessed through key (class).

    Input arguments:
        data:   Data to be processed - list of dictionaries
        key:    Access key to sort the data
    
    Optional arguments:
        one_class_size: Maximal amount of data returned for each class

    Returns:
        list of dictionaries:   Filtered out excessive data from input to achieve
                                the same amount of data for each class
    '''
    # First, sort entries by key
    sorted_data = {}
    for d in tqdm(data, desc="Equalizing entries count", ncols=100):
        if d[key] in sorted_data:
            sorted_data[d[key]].append(d)
        else:
            sorted_data[d[key]] = [d]

    # Requested length could not be larger than len of smallest list
    min_len = min([len(x) for x in sorted_data.values()])
    if one_class_size:
        min_len = min(min_len, one_class_size)

    # Crop each entry of sorted data and shuffle it into 1d array
    # TQDM 'hack' to stop printing the progress bar before the print statement
    time.sleep(0.5)
    rv = np.array([value[:min_len] for value in sorted_data.values()]).flatten()
    print("For each output, " + str(min_len) + " inputs were returned.")
    return rv


def get_data(one_class_size=None, force=True):
    '''
    Get review text - overall ranking dataset for our LSTM network
    
    Optional arguments:
        one_class_size: Amount of requested data for each overall ranking value
        force:          Do not try to load the data from pickle file
    
    Returns:
        list of dictionaries:   Data ready for LSTM network
    '''
    # Generate pickle file path and look, if exists. If exist load it end return int
    path = f"data/amazonDS_{one_class_size}.p" if one_class_size else "amazonDS.p"
    if os.path.exists(path) and not force:
        with open(path, 'rb') as f:
            print("Pickled dataset loaded")
            return pickle.load(f)
    else:
        # If pickle file does not exist, load the data and pickle it
        # Get the whole dataset
        data = _load_data()
        # Pick only entries: overall and reviewText
        data = _filter_json_entries(data, keys=['overall', 'reviewText'])
        # Filter out short data
        data = _filter_short_data(data)
        # Get equal number of reviews for each overall rating
        data = _eliminate_apriori_inequity(data, 'overall', one_class_size=one_class_size)
        # Parse overall from string to int and split review into words
        for d in data:
            d['overall'] = int(d['overall'])
            d['reviewText'] = d['reviewText'].split(" ")
        # Store data as np array in order to be shuffled (they are ordered by class)
        data = np.array(data)
        np.random.shuffle(data)
        # Do not pickle the dataset if forced
        if not force:
            assert_files_safe_dump(path)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    return data


def word_trim_lower(word):
    '''
    Trim the input word and lower its' chars
    '''
    return regex.sub('', word).lower()


def create_embedding_dict(class_key='overall', data_key='reviewText'):
    '''
    Create vector for each word in dataset, each element of vector will
        correspond to probability of occurance in each class.
    
    Pickle created dictionary into path EMBED

    Input arguments:
        class_key:  Key accessing data entry class
        data_key:   Key accessing data dictionary entry training data
    '''
    # Get dataset
    data = get_data(force=True)
    dictionary = {}
    # Get all classes
    classes = set([d[class_key] for d in data])
    # Count words occurences for each class
    word_norm = {c: 0 for c in classes}

    # Count the words for each class
    for review in tqdm(data, desc="Creating embedding dict", ncols=100):
        for word in review[data_key]:
            w = word_trim_lower(word)
            if not w in dictionary:
                dictionary[w] = np.zeros(len(classes))
            # Count, the words, based on rating
            ### This only works for our purpose or if classes are natural nu,bers starting from 1 ###
            dictionary[w][review[class_key]-1] += 1
        word_norm[review[class_key]] += len(review[data_key])
    
    # Normalize vectors
    for word in dictionary.keys():
        dictionary[word] = dictionary[word] / np.array([word_norm[key] for key in sorted(list(word_norm.keys()))])
        dictionary[word] = dictionary[word] / dictionary[word].sum()
    # Save the embedding dictionary
    assert_files_safe_dump(EMBED)
    with open(EMBED, 'wb') as f:
        pickle.dump(dictionary, f)
    del data
