import struct
from PIL import Image, ImageEnhance

ETL1='>H2sH6BI4H4B4x2016s4x'

for fileNum in range(13,14):
    num = str(fileNum)
    if(fileNum < 10): num = '0'+num

    filename = 'ETL1/ETL1C_' + num
    iImage = open(filename, 'rb')

    while(True):
        s = iImage.read(2052)
        if( len(s) != 2052):   break
        r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
        iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
        iP = iF.convert('L')
        fn = "./katakana2/"+str(r[3])+'/'+str(r[0])+"."+str(r[2])+"."+str(r[3])+".jpg"
        enhancer = ImageEnhance.Brightness(iP)
        im = enhancer.enhance(32)
        im.save(fn,"JPEG")

    iImage.close()
