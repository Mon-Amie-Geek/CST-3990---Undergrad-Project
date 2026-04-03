"""
Loads UA-DETRAC ground-truth annotations for Block B evaluation.
Parses the official XML format into per-frame GT dictionaries.
Only reads from official training set sequences (test split of those).
Frame indices are 1-based in UA-DETRAC XML — stored as-is,
     compared against cache frame_idx (0-based) with offset applied in runner.
"""

import os
import xml.etree.ElementTree as ET
# Used because code's shape is : gt_frames[name_num][obj_id] = [x1, y1, x2, y2]
from collections import defaultdict


INCLUDE_OCCLUSION = [0, 1] 

# Function loads GT for one sequence
def load_gt_sequence(xml_path, include_occlusion=INCLUDE_OCCLUSION):
    """
    Parses UA-DETRAC XML annotation for one sequence.

    Returns:
        gt_frames: dict {frame_idx (1-based int): {obj_id (int): [x1,y1,x2,y2]}}
        Example: {1: {1: [100,200,150,250], 2: [300,400,350,450]}, ...}
        Outer keys: frame number, inner keys, object IDs, values = bounding box
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Creates main output structure
    gt_frames = defaultdict(dict)

    # Loops through every frame present in XML: Each frame_elem represents 1 frame in the sequence
    for frame_elem in root.findall('frame'):
        frame_num = int(frame_elem.get('num'))  # 1-based

        # Where annotatted objects for the fame are stored
        target_list = frame_elem.find('target_list')
        if target_list is None:
            continue
        
        # Each target is one labeled vehicle/object
        for target in target_list.findall('target'):
            # Reads the object ID
            obj_id = int(target.get('id'))

            # Where metadata eg: occlusion is stored
            attr = target.find('attribute')
            if attr is not None:
                # If occlusion exists, real value is used. Otherwise, no occlusion is assumed.
                occ = int(attr.get('occlusion', 0))
                if occ not in include_occlusion:
                    continue
                
            # If no bounding box, target is skipped
            box = target.find('box')
            if box is None:
                continue

            left   = float(box.get('left'))
            top    = float(box.get('top'))
            width  = float(box.get('width'))
            height = float(box.get('height'))

            # Standard project format - Many trackers and evaluators use corner coordinates
            x1 = left
            y1 = top
            x2 = left + width
            y2 = top + height

            gt_frames[frame_num][obj_id] = [x1, y1, x2, y2]
            
    # Converts defaultdict into normal dict before returning - Easier to debug, normal dicts are cleaner for later use
    return dict(gt_frames)

# Builds & validates correct XML file path for given sequence ID
def get_gt_xml_path(seq_id, gt_root):
    """
    Resolves path to UA-DETRAC XML for a given sequence ID.
    gt_root: path to datasets/ua_detrac/annotations directory
    """
    xml_path = os.path.join(gt_root, f'{seq_id}.xml')
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"GT XML not found for {seq_id} in {gt_root}. "
            f"Check that DETRAC-Train-Annotations-XML is in place."
        )
    return xml_path