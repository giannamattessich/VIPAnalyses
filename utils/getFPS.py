import xml.etree.ElementTree as ET
import os, numpy as np

def get_fps_from_xml(xml_file):
    if not os.path.exists(xml_file):
        raise ValueError('Experiment.XML file not found to get imaging FPS!!!')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    lsm_info = root.findall('LSM')
    framerate = float(lsm_info[0].get('frameRate'))
    if root.find('LSM').get('averageMode') == '1':
        framerate = float(framerate) / float(root.find('LSM').get('averageNum'))
    return framerate

def get_camera_fps(camera_times, method='median'):
    '''OPTIONS: method = median or method = total'''
    if method == 'median':
        return 1 / np.median(np.diff(camera_times))
    else:
        return len(camera_times) / (camera_times[-1] - camera_times[0])