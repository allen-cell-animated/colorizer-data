

import os
import pandas as pd
import argparse


import json


if __name__ == "__main__":
    ALL_collections = []
    path_to_condition_outputs= "/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/Transcription_factor_cell_lines/Version_1/colorizer_output/EOMES"
    print(os.listdir(path_to_condition_outputs))
    filenames = [os.path.join(path_to_condition_outputs,f) for f in os.listdir(path_to_condition_outputs) if os.path.isdir(os.path.join(path_to_condition_outputs,f))]
    print(filenames)

    for path in filenames:
        movie_entry = {"name": os.path.basename(path).split(".csv",1)[0], "path": os.path.join(os.path.basename(path), "manifest.json")}
        ALL_collections.append(movie_entry)
        assert os.path.exists(os.path.join(path, "manifest.json"))

    # ALL_collections.append(["", ])
    out_file = open("/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/Transcription_factor_cell_lines/Version_1/colorizer_output/EOMES/collections_all.json", "w")
    print(ALL_collections)
    json.dump(ALL_collections, out_file) 

    #/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/Version_5/colorizer_output/all_conditions



