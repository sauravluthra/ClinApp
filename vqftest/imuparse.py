import pandas as pd
from vqf import VQF, BasicVQF, PyVQF
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import remove_frame

#***********************************************
def main():

    data = fetchData("imudata/imu123.txt")

    #gyr = np.deg2rad(1)*np.ones((10, 3))  # in rad/s
    #ryg = np.ascontiguousarray(data[['gx', 'gy', 'gz']].to_numpy())

    #print(str(gyr) + "\n\n==========\n\n" + str(ryg))

    for id, imuData in enumerate(data):
        runvqf(imuData, (id))

    print("\n\nEXIT PROGRAM")

    return 0

def runvqf(data, imuID):

    # generate simple dummy data
    #gyr = np.deg2rad(1)*np.ones((6000, 3))  # in rad/s
    #acc = 9.81/np.sqrt(3)*np.ones((6000, 3))  # in m/s²
    gyr = np.ascontiguousarray(data[['gx', 'gy', 'gz']].to_numpy())
    acc = np.ascontiguousarray(data[['ax', 'ay', 'az']].to_numpy())
    mag = np.ascontiguousarray(data[['mx', 'my', 'mz']].to_numpy())
    #print("GYR: \n" + str(gyr))
    Ts = 0.01  # sampling time (100 Hz)

    # run orientation estimation
    vqf = VQF(Ts)
    # alternative: vqf = PyVQF(Ts)
    out = vqf.updateBatch(gyr, acc)

    print(f"\n\nIMU {imuID} OUTPUT\n\n" + str(out))

    #with open(f"Quaternion_Out_IMU{imuID}", "w") as f:
       # f.write(str(out))

    # plot the quaternion
    plt.figure()
    plt.subplot(211)
    plt.plot(out['quat6D'])
    plt.title('quaternion 6D with Bias Est. | IMU#' + str(imuID))
    plt.grid()

    # run the basic version with the same data
    #params = dict(
    #    motionBiasEstEnabled=False,
    #    restBiasEstEnabled=False,
    #    magDistRejectionEnabled=False,
    #)
    #vqf2 = VQF(Ts, **params)
    vqf2 = VQF(Ts)
    # alternative: vqf2 = BasicVQF(Ts)
    # alternative: vqf2 = PyVQF(Ts, **params)
    out2 = vqf2.updateBatch(gyr, acc, mag)

    # plot quaternion (notice the difference due to disabled bias estimation)
    plt.subplot(212)
    plt.plot(out2['quat9D'])
    plt.grid()
    plt.title('quaternion 9D with Bias Est.  | IMU#' + str(imuID))
    plt.tight_layout()
    plt.show()

    np.set_printoptions(threshold=np.inf)

    with open(f"output/QuatOutput6D_{imuID}.txt", "w+") as f:
        f.write(str(out['quat6D']))

    with open(f"output/QuatOutput9D_{imuID}.txt", "w+") as f:
        f.write(str(out2['quat9D']))


def visualizeEuler():
    return 0

def fetchData(fileName):

    #Number of IMU sensors
    IMU_COUNT = 3

    #Create a list of dataframes representing each IMU sensor's data
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    #df4 = pd.DataFrame()
    #df5 = pd.DataFrame()

    dfArr = [df1, df2, df3]
    
    with open(fileName, "r") as f:

        #Full number of samples among all the IMUs
        fullTxt = f.readlines()
        numSamples = len(fullTxt)//5
        print("NUMSAMPLES: " + str(numSamples))

        for i in range (0, numSamples):

            try:
                sample = fullTxt[i * 5 : i * 5 + 5]

                imuID     = int(sample[0].strip("IMU #:"))
                timestamp = int(sample[1])

                a = sample[2].strip("\n \t").split(",")
                g = sample[3].strip("\n \t").split(",")
                m = sample[4].strip("\n \t").split(",")

                new_row = pd.DataFrame()
                new_row = { "milliseconds": timestamp, 
                            "gx": np.deg2rad(float(g[0])),
                            "gy": np.deg2rad(float(g[1])),
                            "gz": np.deg2rad(float(g[2])),
                            #"gx": float(g[0]),
                            #"gy": float(g[1]),
                            #"gz": float(g[2]),
                            "ax": float(a[0]),
                            "ay": float(a[1]),
                            "az": float(a[2]),
                            "mx": float(m[0]),
                            "my": float(m[1]),
                            "mz": float(m[2]),
                        }
                dfArr[imuID] = dfArr[imuID].append(new_row, ignore_index=True)
            except:
                print(f"ERROR OCCURED: \n{str(sample)}")
                break

    print("Succesfully Fetched Data from " + fileName + " (" + str(numSamples) + " samples)\n")
    #print("\n\nIMU 0\n\n" + str(dfArr[0].head(3)))
    #print("\n\nIMU 1\n\n" + str(dfArr[1].head(3)))
    #print("\n\nIMU 2\n\n" + str(dfArr[1].head(3)))
    #print("\n\nIMU 3\n\n" + str(dfArr[1].head(3)))
    #print("\n\nIMU 4\n\n" + str(dfArr[1].head(3)))

    return dfArr

if __name__ == '__main__':
    main()