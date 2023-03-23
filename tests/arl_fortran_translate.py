import numpy as np
import struct
import os, sys

# function to unpack data
def unpack(cpack, nx, ny, nexp, var1):
    rdata = np.empty((nx, ny), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            idata = (j * nx) + i
            kpack = idata + 1
            int1 = ord(cpack[kpack])
            if int1 < 128:
                rdata[i, j] = float(int1 + 128 * (int1 < 64))
            else:
                int2 = ord(cpack[kpack + 1])
                if int1 < 192:
                    rdata[i, j] = float(((int1 - 128) << 8) + int2 + 32768)
                else:
                    int3 = ord(cpack[kpack + 2])
                    if int1 < 224:
                        rdata[i, j] = float(((int1 - 192) << 16) + (int2 << 8) + int3 + 8388608)
                    else:
                        int4 = ord(cpack[kpack + 3])
                        rdata[i, j] = float(((int1 - 224) << 24) + (int2 << 16) + (int3 << 8) + int4)
    return rdata

# main program
if __name__ == '__main__':
    # read input
    fdir = input('Enter directory name: ').strip()
    file = input('Enter file name: ').strip()

    # test for file existence
    if not os.path.exists(os.path.join(fdir, file)):
        print(f'Unable to find file: {file}')
        print(f'On local directory: {fdir}')
        sys.exit()

    # open file to decode the standard label (50) plus the fixed portion (108) of the extended header
    with open(os.path.join(fdir, file), 'rb') as f:
        header = f.read(158)
        iy, im, id, ihr, ifc, kvar = struct.unpack('5H4x4s', header[:16])
        iy = iy - 1900 if iy > 1900 else iy + 100
        print(f'Opened file: {iy:4}{im:4}{id:4}{ihr:4}')
        if kvar != b'INDX':
            print('WARNING Old format meteo data grid')
            print(f'{iy:4}{im:4}{id:4}{ihr:4}  {kvar.decode()}')
            sys.exit()

        # decode extended portion of the header
        values = struct.unpack('4s3H9f3H3x', header[16:])
        model, icx, mn, pole_lat, pole_lon, ref_lat, ref_lon, size, orient, tang_lat, sync_xp, sync_yp, \
            sync_lat, sync_lon, dummy, nx, ny, nz, k_flag, lenh = values

    # close file and reopen with proper length
    len = nx * ny + 50
    with open(os.path.join(fdir, file), 'rb') as f:
        f.seek(0, 2)
        if f.tell() < len:
            print('File is not complete')
            sys.exit()
        f.seek(0)

        # print file diagnostic
        print(f'Grid size and lrec: {nx} {ny}')
