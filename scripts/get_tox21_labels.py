#!/usr/bin/env python3
"""
Get list of available Tox21 assay labels.
"""

from tdc.utils import retrieve_label_name_list

if __name__ == "__main__":
    labels = retrieve_label_name_list("Tox21")
    print(f"Available Tox21 assays ({len(labels)}):")
    for label in labels:
        print(f"  - {label}")
