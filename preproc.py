# Utilizes for loading raw data

import struct
import numpy as np
from PIL import Image, ImageEnhance


ETL_PATH = 'C:/Users/G-ICT/ETLdatasets'

def read_record(database, f):
    W, H = 64, 63
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (W,H), r[18], 'bit', 4)
    iP = iF.convert('P')

    enhancer = ImageEnhance.Brightness(iP)
    iE = enhancer.enhance(40)
    size_add = 12

    iE = iE.resize((W + size_add, H + size_add))
    iE = iE.crop((size_add / 2,
                  size_add / 2,
                  W + size_add / 2,
                  H + size_add / 2))
    img_out = r + (iE,)
    return img_out

def get_ETL_data(dataset, categories, writers_per_char,
                 database='ETL1C',
                 starting_writer=None,
                 vectorize=False,
                 resize=None,
                 img_format=False,
                 get_scripts=False):

    W, H = 64, 64
    new_img = Image.new('1', (W,H))

    name_base = ETL_PATH + '/ETL1/ETL1C_'

    try:
        filename = name_base + dataset
    except:
        filename = name_base + str(dataset)

    X = []
    Y = []
    scriptTypes = []

    try:
        iter(categories)
    except:
        categories = [categories]

    for id_category in categories:
        with open(filename, 'r') as f:
            f.seek((id_category * 1411 + 1) * 2052)

            for i in range(writers_per_char):
                try:
                    # skip records
                    if starting_writer:
                        for j in range(starting_writer):
                            read_record(database, f)

                    # start outputting records
                    r = read_record(database, f)
                    new_img.paste(r[-1], (0,0))
                    iI = Image.eval(new_img, lambda x: not x)

                    # resize images
                    if resize:
                        # new_img.thumbnail(resize, Image.ANTIALIAS)
                        iI.thunmbnail(resize)
                        shapes = resize[0], resize[1]

                    else:
                        shape = W, H

                    # output formats
                    if img_format:
                        outData = iI
                    elif vectorize:
                        outData = np.asanyarray(iI.getdata()).reshape( shapes[0] * shapes[1])
                    else:
                        outData = np.asarray(iI.getdata()).reshape( shapes[0] * shapes[1])

                    X.append(outData)
                    Y.append(r[3])
                    scriptTypes.append(1)
                except:
                    break

    output = []
    if img_format:
        output += [X]
        output += [Y]
    else:
        X, Y = np.asarray(X, dtype=np.int32), np.asanyarray(Y, dtype=np.int32)
        output += [X]
        output += [Y]

    if get_scripts:
        output += [scriptTypes]


    return output

get_ETL_data('01')