import json
import csv
import os
import glob

from IPython.core.debugger import set_trace
def get_all_experiments(folder, folder_keys, fileout):
    set_trace()
    folder_keys =   os.path.join(folder_keys, 'config_train.json')
    assert os.path.isfile(folder_keys), 'Key folder does not exists'

    with open(folder_keys) as fp:
        config  =   json.load(fp)
    
    keys    =   list(config.keys())

    folders_ = os.path.join(folder, 'sample')
    folders_ = glob.glob(folders_ + '*')

    config_files    =   [os.path.join(folder_, 'config_train.json') for folder_ in folders_]
    
    config_dicts    =   []
    for config_file in config_files:
        with open(config_file) as fp:
            config_dicts.append(json.load(fp))
    def getid(ll):
        return int(ll['id_executor'][6:])
    config_dicts.sort(key=getid)
    #config_dict =   [with open(config_file) as fp in ]
    with open(fileout+'.csv', 'w') as csvfile:
        fields = keys
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(config_dicts)
    x=21
get_all_experiments('./data','./data/sample41','experiments')
