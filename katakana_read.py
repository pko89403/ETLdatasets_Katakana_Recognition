import struct
from PIL import Image, ImageEnhance

ETL1='>H2sH6BI4H4B4x2016s4x'

for fileNum in range(7,8):
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
        fn = "./katakatna/"+str(r[0])+"."+str(r[2])+"."+str(r[3])+".bmp"
        enhancer = ImageEnhance.Brightness(iP)
        gray = enhancer.enhance(16)
        bw = gray.point(lambda x : 0 if x < 128 else 255, '1')
        bw.save(fn,'BMP')
        bw.close()

    iImage.close()
