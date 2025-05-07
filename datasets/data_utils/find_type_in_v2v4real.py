import os
import numpy
from collections import OrderedDict
from hypes_yaml.yaml_utils import load_yaml

# train: ['Car', 'Truck', 'Van', 'ConcreteTruck', 'Pedestrian', 'Bus', 'ScooterRider']
# test: ['Car', 'Truck', 'Van', 'ConcreteTruck', 'Pedestrian', 'BicycleRider', 'Scooter', 'Bus']
root_dir = '/data1/dataset_zhiying/V2V4Real/test'
validate_dir = '/data1/dataset_zhiying/V2V4Real/test'
max_cav = 2

# first load all paths of different scenarios
scenario_folders = sorted([os.path.join(root_dir, x)
                            for x in os.listdir(root_dir) if
                            os.path.isdir(os.path.join(root_dir, x))])


# Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
# lidar: path, cameras:list of path}}}}        
scenario_database = OrderedDict()
len_record = []

def extract_timestamps(yaml_files):
    """
    Given the list of the yaml files, extract the mocked timestamps.

    Parameters
    ----------
    yaml_files : list
        The full path of all yaml files of ego vehicle

    Returns
    -------
    timestamps : list
        The list containing timestamps only.
    """
    timestamps = []

    for file in yaml_files:
        res = file.split('/')[-1]

        timestamp = res.replace('.yaml', '')
        timestamps.append(timestamp)

    return timestamps

types = []
# loop over all scenarios
for (i, scenario_folder) in enumerate(scenario_folders):
    scenario_database.update({i: OrderedDict()})

    # at least 1 cav should show up
    cav_list = sorted([x for x in os.listdir(scenario_folder)
                        if os.path.isdir(
            os.path.join(scenario_folder, x))])
    assert len(cav_list) > 0

    # roadside unit data's id is always negative, so here we want to
    # make sure they will be in the end of the list as they shouldn't
    # be ego vehicle.
    if int(cav_list[0]) < 0:
        cav_list = cav_list[1:] + [cav_list[0]]

    # loop over all CAV data
    for (j, cav_id) in enumerate(cav_list):
        if j > max_cav - 1:
            print('too many cavs')
            break
        scenario_database[i][cav_id] = OrderedDict()

        # save all yaml files to the dictionary
        cav_path = os.path.join(scenario_folder, cav_id)

        # use the frame number as key, the full path as the values
        yaml_files = \
            sorted([os.path.join(cav_path, x)
                    for x in os.listdir(cav_path) if
                    x.endswith('.yaml') and 'additional' not in x])
        timestamps = extract_timestamps(yaml_files)

        for timestamp in timestamps:
            scenario_database[i][cav_id][timestamp] = \
                OrderedDict()

            yaml_file = os.path.join(cav_path,
                                        timestamp + '.yaml')

            params = load_yaml(yaml_file)
            for id, object in params['vehicles'].items():
                typ = object['obj_type']
                if not typ in types:
                    types.append(typ)
            
print(types)
pass