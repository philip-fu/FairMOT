import json
import fire
import os
import tqdm
import subprocess
from collections import defaultdict
from google.cloud import bigquery, storage

BQ_CLIENT = bigquery.Client() # run `gcloud auth application-default login` to log in at least for the first time
STORAGE_CLIENT = storage.Client()

def download_images_from_gs(image_prefix, image_extension, image_folder):
    download_command = "gsutil -m -o GSUtil:parallel_process_count=1 -o GSUtil:parallel_thread_count=24 cp {}*{} {}".format(image_prefix, image_extension, image_folder)
    subprocess.run(download_command, shell=True, stderr=subprocess.STDOUT)

    return True

def rename_images(image_folder, image_extension):
    rename_command = "cd {}; for f in *{}; do   mv $f ".format(image_folder, image_extension) + "${f//frame_/0}; done"
    subprocess.run(rename_command, shell=True, stderr=subprocess.STDOUT, executable='/bin/bash')

    return True


def convert_images(image_folder, image_extension):
    conver_command = f"cd {image_folder}; mogrify -format jpg *{image_extension} && rm *{image_extension}"
    subprocess.run(conver_command, shell=True, stderr=subprocess.STDOUT, executable='/bin/bash')

    return True


def parse_gt_json(directory_name, output_base_dir='dataset/ap/images/train',
                  frame_rate=12, image_w=1920, image_h=1080,
                  image_folder='img1', img_extension='.png'):
    """Get GT from big query table and save as MOT format
    """

    seq_name = directory_name
    output_base_dir = os.path.join(output_base_dir, seq_name)
    image_folder_full = os.path.join(output_base_dir, image_folder)
    os.makedirs(os.path.join(output_base_dir, 'det'), exist_ok = True)
    os.makedirs(os.path.join(output_base_dir, 'gt'), exist_ok = True)
    os.makedirs(image_folder_full, exist_ok = True)

    
    query = ("""
        SELECT * FROM `wmt-892b8050ebef7b873b51427a84.ai_platform_prod.kepler_scrappie_vendors_direct` 
        where directory_name in ('{}')
        order by xml_path, ingest_datetime desc
    """.format(directory_name))

    print(query)
    query_job = BQ_CLIENT.query(query)  # API request
    bq_rows = query_job.result()

    image_uri_dict = {}

    # write detections
    with open(os.path.join(output_base_dir, 'gt/gt.txt'), 'w') as f:
        # this is for items
        pivot = 1
        for bq_row in bq_rows:
            if bq_row.get('xml_path') in image_uri_dict:
                continue
            image_uri_dict[bq_row.get('xml_path')] = bq_row.get('image_path')
            for det in bq_row.get('objects'):
                if det['class'] == 'item':
                    frame_id = pivot # should start with 1
                    tracking_id = det['id']
                    bb_left = det['bndbox']['xmin']
                    bb_top = det['bndbox']['ymin']
                    bb_width = det['bndbox']['xmax'] - det['bndbox']['xmin']
                    bb_height = det['bndbox']['ymax'] - det['bndbox']['ymin']
                    conf = 1

                    f.write('{},{},{},{},{},{},{},-1,-1,-1\n'.format(
                        frame_id,
                        tracking_id,
                        bb_left,
                        bb_top,
                        bb_width,
                        bb_height,
                        conf
                    ))
            pivot += 1

        

    if len(image_uri_dict) > 0:
        print("Downloading images..")
        image_name = image_uri_dict[next(iter(image_uri_dict))]
        image_prefix = image_name.replace(image_name.split('/')[-1], '')
        print(image_prefix, image_folder_full)
        download_images_from_gs(image_prefix.replace('/annotations', '/images'), img_extension, image_folder_full)
        
        # rename and change to jpg
        rename_images(image_extension=img_extension, image_folder=image_folder_full)
        convert_images(image_extension=img_extension, image_folder=image_folder_full)
        
        

    # write a sequence info file
    seq_info_filename='seqinfo.ini'
    with open(os.path.join(output_base_dir, seq_info_filename), 'w') as f:
        f.write('[Sequence]\n')
        f.write('name={}\n'.format(seq_name))
        f.write('imDir={}\n'.format(image_folder))
        f.write('frameRate={}\n'.format(frame_rate))
        f.write('seqLength={}\n'.format(pivot-1))
        f.write('imWidth={}\n'.format(image_w))
        f.write('imHeight={}\n'.format(image_h))
        f.write('imExt={}\n'.format(img_extension))

    return True



def get_ap_dataset_from_bq(output_base_dir='dataset/ap/images/train',
                           frame_rate=12, image_w=1920, image_h=1080,
                           image_folder='img1', img_extension='.png'):
    query = ('''
        SELECT 
        directory_name
        FROM 
        (
            SELECT *, CAST(REGEXP_EXTRACT(image_path, r"frame_([^/$]*)\.{1}"''' + """) AS INT64) AS frame_index
            FROM ai_platform_prod.kepler_scrappie_vendors_direct
            where image_path like '%{}'
        ) f
        group by directory_name
        having max(frame_index) - min(frame_index) + 1 = count(distinct image_path)
    """.format(img_extension))

    print(query)
    query_job = BQ_CLIENT.query(query)  # API request
    bq_rows = query_job.result()
    
    for bq_row in tqdm.tqdm(bq_rows):
        directory_name = bq_row.get('directory_name')
        print(directory_name)
        parse_gt_json(directory_name, output_base_dir=output_base_dir,
                      frame_rate=frame_rate, image_w=image_w, image_h=image_h,
                      image_folder=image_folder, img_extension=img_extension)



def get_store_lane_from_bq(store_lane_pairs, coordinate_json_file):
    """Query from BQ in GCP for store-lane coordinates. 
       Need to be on VPN?
    
    Args:
        store_lane_pairs: A string of <store_id>-<lane_id>s, separated by ','. E.g., 4184-34,1440-33
        coordinate_json_file: The json file to update to.
    """

    store_lane_pairs = store_lane_pairs.split(',')

    query = ("""
        SELECT * FROM `wmt-892b8050ebef7b873b51427a84.azure_datasets_prod.ap_verify_sco_coordinates_response` 
        where concat(store_id, '-', lane_id) in ({})
    """.format(str(store_lane_pairs)[1:-1]))

    print(query)
    query_job = BQ_CLIENT.query(query)  # API request
    bq_rows = query_job.result()  # Waits for query to finish
    
    json_data = defaultdict(lambda: {})
    if os.path.isfile(coordinate_json_file): 
        with open(coordinate_json_file, 'r') as f:
            json_data.update(json.load(f))
    
    found = []
    for bq_row in tqdm.tqdm(bq_rows):
        row = bq_fieldname_to_config_fieldname(bq_row)
        lane_data = {'lane-' + str(bq_row.get('lane_id')): {'coordinates': row}}
        json_data[bq_row.get('store_id')].update(lane_data)
        found.append(str(bq_row.get('store_id')) + '-' + str(bq_row.get('lane_id')))
        
    with open(coordinate_json_file, 'w') as f:
        json.dump(json_data, f)

    for store_lane_pair in store_lane_pairs:
        if store_lane_pair not in found:
            print("{} is not found".format(store_lane_pair))

    return True

def map_coord_format(bq_coord):
    return str(bq_coord).replace('xmin', 'xMin').replace('xmax', 'xMax').replace('ymin', 'yMin').replace('ymax', 'yMax')
    


def bq_fieldname_to_config_fieldname(bq_row):
    m = {
            'coordinates_adjusted_scanner': 'scanner',
            'coordinates_adjusted_scanner_bed': 'scanner_bed',
            'coordinates_adjusted_scanner_light': 'scanner_light',
            'coordinates_adjusted_bagging': 'bagging_area',
            'coordinates_adjusted_screen': 'screen_area',
            'coordinates_adjusted_head_hand_scanner': 'head_hand_scanner_area',
            'coordinates_adjusted_mount_hand_scanner': 'mount_hand_scanner_area'
        }
    
    row = dict(bq_row)
    row = {m[k]:map_coord_format(row[k]) for k in row if k in m}
    row['scanner_bed_ts'] = row['scanner_bed']

    return row


if __name__ == '__main__':
    fire.Fire()