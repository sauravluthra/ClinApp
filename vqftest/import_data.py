import numpy as np
import pandas as pd
from vqf import VQF, offlineVQF
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import remove_frame

#***********************************************
def main():
    data = fetchData("imudata/imu 45 2.txt")

    for id, imuData in enumerate(data):
        runvqf(imuData, (id))

    print("\n\nEXIT PROGRAM")
    return 0

def runvqf(data, imuID):

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

    plt.subplot(212)
    plt.plot(vqf['quat9D'])
    plt.grid()
    plt.title('quaternion 9D with Bias Est.  | IMU#' + str(imuID))
    plt.tight_layout()
    plt.show()

    np.set_printoptions(threshold=np.inf)


    with open(f"output/QuatOutput9D_{imuID}.csv", "w+"):
        np.savetxt(f"output/QuatOutput9D_{imuID}.csv", vqf['quat9D'], delimiter = ",")


def fetchData(fileName):
    # Set axes to account for sensor orientation
    x = 2
    y = 0
    z = 1

    # Number of IMU sensors
    IMU_COUNT = 5

    # Create a list of dataframes representing each IMU sensor's data
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df6 = pd.DataFrame()

    dfArr = [df1, df2, df3, df4, df5, df6]
    
    with open(fileName, "r") as f:

        # Full number of samples among all the IMUs
        fullTxt = f.readlines()
        numSamples = len(fullTxt) // 5
        print("NUMSAMPLES: " + str(numSamples))

        for i in range (0, numSamples):

            try:
                sample = fullTxt[i * 5 : i * 5 + 5] # Read 5 lines

                imuID     = int(sample[0].strip("IMU #:"))
                timestamp = int(sample[1])

                a = sample[2].strip("\n \t").split(",")
                g = sample[3].strip("\n \t").split(",")
                m = sample[4].strip("\n \t").split(",")

                new_row = pd.DataFrame()
                new_row = { "milliseconds": timestamp, 
                            "gx": np.deg2rad(float(g[x])),
                            "gy": np.deg2rad(float(g[y])),
                            "gz": np.deg2rad(float(g[z])),
                            "ax": float(a[x]),
                            "ay": float(a[y]),
                            "az": float(a[z]),
                            "mx": float(m[x]),
                            "my": float(m[y]),
                            "mz": float(m[z]),
                        }
                dfArr[imuID] = dfArr[imuID].append(new_row, ignore_index=True)
            except:
                print(f"ERROR OCCURED: \n{str(sample)}")
                break

    print("Succesfully Fetched Data from " + fileName + " (" + str(numSamples) + " samples)\n")
    
    for d in dfArr:
        print(f"datadatadata\n\n{str(d)}")

    return dfArr

if __name__ == '__main__':
    main()