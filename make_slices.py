import argparse
from glob import glob
import os
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

def read_coreplug(
  raw,
  slice_width,
  slice_height,
  slices_num,
  raw_type = 'uint8',
  little_endian = True,
  save_type = 'uint8'):

  slices = tf.io.decode_raw(tf.io.read_file(raw), raw_type, little_endian=little_endian)
  slices = tf.reshape(slices, [slice_width, slice_height, slices_num])
  slices = tf.cast(tf.transpose(slices, [2, 0, 1]), save_type)
  slices = tf.unstack(slices)

  return slices

  
def gen_slises(
    input_dir,
    slice_width,
    slice_height,
    slices_num,
    output_dir,
    raw_type = 'uint8',
    little_endian = True,
    save_type = 'uint8'
):
    raws_files_list = glob(os.path.join(input_dir, '*.raw'))
    raws_files_list_len = len(raws_files_list)
    for j, raw in enumerate(raws_files_list):
        #f = open(raw, 'rb')
        print(f'Slice #{j+1}/{raws_files_list_len}')
        dirname = os.path.splitext(os.path.basename(raw))[0]
        slices = read_coreplug(
          raw,
          slice_width,
          slice_height,
          slices_num,
          raw_type,
          little_endian,
          save_type
        )
        
        output_folder_path = os.path.join(output_dir, dirname)
        for i,s in enumerate(tqdm(slices, desc='Slices')):
            Path(output_folder_path).mkdir(parents=True, exist_ok=True)
            tf.io.write_file(
                os.path.join(output_folder_path, f'slice_{i:05d}.slice'), 
                tf.io.serialize_tensor(s))

def main():
    
    sup_types = [
        'uint8', 'uint16', 'uint32', 'uint64',
        'int8', 'int16', 'int32', 'int64',
        'float16', 'float32', 'float64'
    ]
    
    parser = argparse.ArgumentParser(description='Generates NPY slices from RAW files')
    parser.add_argument("-i", "--input", help="input directory with raw files")
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument('--width', type=int, default=1000, help="slice width")
    parser.add_argument('--height', type=int, default=1000, help="slice height")
    parser.add_argument('--slices', type=int, default=1000, help="number of slices")
    parser.add_argument('-t', '--type', choices=sup_types, default='uint8', help="image type")
    parser.add_argument('-st', '--save_type', choices=sup_types, default='uint8', help="slice type for saving")
    parser.add_argument('-l', '--little_endian', type=bool, default=True, help="little-endian byte order")
    
    args = parser.parse_args()
    
    gen_slises(
        args.input, args.width, args.height, 
        args.slices, args.output, args.type, args.little_endian, args.save_type)

if __name__ == "__main__":
    main()