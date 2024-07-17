import numpy as np
import pandas as pd
from vqf import VQF, offlineVQF
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import remove_frame
import os
import base64 as b64

#***********************************************
def main():
    data = fetchData("imudata/LIAN3/static2.txt")
    dirName = "LIAN3/static2"
    for id, imuData in enumerate(data):
        runvqf(imuData, (id), dirName)

    print("\n\nEXIT PROGRAM")
    return 0

def runvqf(data, imuID, dirName):

    try:
        gyr = np.ascontiguousarray(data[['gx', 'gy', 'gz']].to_numpy())
        acc = np.ascontiguousarray(data[['ax', 'ay', 'az']].to_numpy())
        mag = np.ascontiguousarray(data[['mx', 'my', 'mz']].to_numpy())
    except:
        print("ERROR: CANT CONVERT TO NP CONTIGUOUS ARRAY")
        return

    Ts = 0.01  # sampling time (100 Hz)

    # run orientation estimation
    vqf = offlineVQF(gyr, acc, mag,Ts)
    #out_quat = vqf.updateBatch()
    #out_rest = vqf.getRestDetected()
    vqf6D = VQF(Ts)
    out6D = vqf6D.updateBatch(gyr, acc)
    plt.figure()
    plt.subplot(211)
    plt.plot(out6D['quat6D'])
    plt.title(f"QUATERNION 6D w/BIAS ESTIMATE | IMU#{str(imuID)} | {dirName}")
    plt.grid()
    plt.subplot(212)
    plt.plot(vqf['quat9D'])
    plt.grid()
    plt.title(f"QUATERNION 9D w/BIAS ESTIMATE | IMU#{str(imuID)} | {dirName}")
    plt.tight_layout()

    if not os.path.isdir(f"output/{dirName}"):
        os.makedirs(f"output/{dirName}")

    #Save plot as an image and display it
    plt.savefig(os.path.join("output/", dirName, f'Quat9D_IMU{imuID}.png'))
    plt.show()

    np.set_printoptions(threshold=np.inf)

    with open(f"output/{dirName}/QuatOutput6D_{imuID}.csv", "w+"):
        np.savetxt(f"output/{dirName}/QuatOutput6D_{imuID}.csv", out6D['quat6D'], delimiter = ",")
    
    with open(f"output/{dirName}/QuatOutput9D_{imuID}.csv", "w+"):
        np.savetxt(f"output/{dirName}/QuatOutput9D_{imuID}.csv", vqf['quat9D'], delimiter = ",")


def fetchData(fileName):

    # Set axes to account for sensor orientation
    x = 2
    y = 0
    z = 1

    # Create a list of dataframes representing each IMU sensor's data
    df0 = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df6 = pd.DataFrame()

    dfArr = [df0, df1, df2, df3, df4, df5, df6]
    
    with open(fileName, "r") as f:
        
        for line in f:
            line = line.strip("\n")

            try:
                if len(line) == 1:
                    imuID = int(line)
                elif len(line) > 1:
                    lineArr = line.split(" ")

                    for element in lineArr:
                        if element in ["1", "2", "3"]:
                            agm = int(element)
                        else:
                            reading = [b64.b64decode(element[0:2]), b64.b64decode(element[2:4]), b64.b64decode(element[4:6])]

                print(reading)

                new_row = pd.DataFrame()
                new_row = { 
                            "gx": (float(g[x])),
                            "gy": (float(g[y])),
                            "gz": (float(g[z])),
                            "ax": float(a[x]),
                            "ay": float(a[y]),
                            "az": float(a[z]),
                            "mx": float(m[x]),
                            "my": float(m[y]),
                            "mz": float(m[z]),
                        }
                dfArr[imuID] = dfArr[imuID].append(new_row, ignore_index=True)
            except:
                print(f"ERROR OCCURED: \n{str(line)}")
                break

    print("Succesfully Fetched Data from " + fileName)
    
    for d in dfArr:
        print(f"datadatadata\n\n{str(d)}")

    return dfArr

if __name__ == '__main__':
    main()