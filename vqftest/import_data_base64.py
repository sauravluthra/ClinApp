import numpy as np
import pandas as pd
from vqf import VQF, offlineVQF
import matplotlib.pyplot as plt
import os
import base64
import math

#***********************************************
def main():

    df = pd.DataFrame({'gx':[1,2,3,4,5,6,7,8,9], 'gy':[1,2,3,4,5,6,7,8,9], 'gz':[1,2,3,4,5,6,7,8,9]})

    data = fetchData("b64.txt")
    dirName = "b64out/"

    print(str(data))
    for key in data:
        print(str(key) + '\n' + str(data[key].fillna(0)))
        runvqf(data[key].fillna(0), key, dirName)

    print("\n\nEXIT PROGRAM")
    return 0

def runvqf(data: pd.DataFrame, imuID: str, dirName: str) -> None:

    try:
        gyr = np.ascontiguousarray(data[['gyrx', 'gyry', 'gyrz']].to_numpy())
        acc = np.ascontiguousarray(data[['accx', 'accy', 'accz']].to_numpy())
        mag = np.ascontiguousarray(data[['magx', 'magy', 'magz']].to_numpy())
    except:
        print("ERROR: CANT CONVERT TO NP CONTIGUOUS ARRAY")
        return

    # sampling time (100 Hz)
    Ts = 0.01  

    # run orientation estimation
    vqf = offlineVQF(gyr, acc, mag,Ts)
    #out_quat = vqf.updateBatch()
    #out_rest = vqf.getRestDetected()
    vqf6D = VQF(Ts)
    out6D = vqf6D.updateBatch(gyr, acc)
    plt.figure()
    plt.subplot(211)
    plt.plot(out6D['quat6D'])
    plt.title(f"QUATERNION 6D w/BIAS ESTIMATE | {(imuID)} | {dirName}")
    plt.grid()
    plt.subplot(212)
    plt.plot(vqf['quat9D'])
    plt.grid()
    plt.title(f"QUATERNION 9D w/BIAS ESTIMATE | {(imuID)} | {dirName}")
    plt.tight_layout()

    if not os.path.isdir(f"output/{dirName}"):
        os.makedirs(f"output/{dirName}")

    #Save plot as an image and display it
    plt.savefig(os.path.join("output/", dirName, f'Quat9D_{imuID}.png'))
    plt.show()

    np.set_printoptions(threshold=np.inf)

    with open(f"output/{dirName}/QuatOutput6D_{imuID}.csv", "w+"):
        np.savetxt(f"output/{dirName}/QuatOutput6D_{imuID}.csv", out6D['quat6D'], delimiter = ",")
    
    with open(f"output/{dirName}/QuatOutput9D_{imuID}.csv", "w+"):
        np.savetxt(f"output/{dirName}/QuatOutput9D_{imuID}.csv", vqf['quat9D'], delimiter = ",")

#Take in a 6-character base64 encoding and convert it to the corresponding X,Y,Z components (array)
def getXYZ(element: str) -> list[int]:
    
    x = element[0:2]
    y = element[2:4]
    z = element[4:6]

    xb = base64.b64decode(x + '==')  
    yb = base64.b64decode(y + '==')  
    zb = base64.b64decode(z + '==')

    x_string = ''.join(f"{byte:08b}" for byte in xb)
    y_string = ''.join(f"{byte:08b}" for byte in yb)
    z_string = ''.join(f"{byte:08b}" for byte in zb)

    x_string1 = x_string[:12]
    y_string1 = y_string[:12]
    z_string1 = z_string[:12]

    xd = int(x_string1, 2)
    yd = int(y_string1, 2)
    zd = int(z_string1, 2)

    #gravity = math.sqrt(xd*xd + yd*yd + zd*zd)

    """print(f"Base64 string: {base64_string}")
    print(f"Decoded bit string: {bit_string}")"""
    #print(f"gravity: {gravity}")

    print(f"{x} => {xb} => {x_string} => {x_string1} => {xd}")
    print(f"{y} => {yb} => {y_string} => {y_string1} => {yd}")
    print(f"{z} => {zb} => {z_string} => {z_string1} => {zd}")

    return [xd, yd, zd]

def fetchData(fileName: str) -> dict[str, pd.DataFrame]:

    # Set axes indexes to account for sensor orientation (0, 1, 2)
    # Which axis (according to the sensor data) which will map to which axis (data fed into vqf)
    x_idx = 2
    y_idx = 0
    z_idx = 1

    #array with IMU sensor names 1-Acc, 2-Gyr, 3-Mag
    sensornames = ["xxx", "acc", 'gyr', 'mag']

    # Create a dict of 21 dataframes representing each IMU's acc, gyr, mag sensors data
    sensors = ["imu0_acc", "imu0_gyr","imu0_mag",
               "imu1_acc", "imu1_gyr","imu1_mag",
               "imu2_acc", "imu2_gyr","imu2_mag",
               "imu3_acc", "imu3_gyr","imu3_mag",
               "imu4_acc", "imu4_gyr","imu4_mag",
               "imu5_acc", "imu5_gyr","imu5_mag",
               "imu6_acc", "imu6_gyr","imu6_mag",
               "imu7_acc", "imu7_gyr","imu7_mag"] 
    data = {}
    for name in sensors:
        data[name] = pd.DataFrame()

    #Open b64 encoded file and iterate through every line
    with open(fileName, "r") as f:
        lines = f.readlines()

        for line in lines:
            #Remove unnecessary characters/whitespace
            line = line.strip("\n\xa0")
            
            #if Line is IMU ID
            if len(line) == 1:
                imuID = "imu" + line + "_"
            #if Line is DATA
            elif len(line) > 1:

                #Split line into an array of elements
                lineArr = line.split(" ")
                
                for element in lineArr:
                    
                    #If element denotes the sensor type 
                    #1-Acc, 2-Gyr, 3-Mag
                    if element in ["1", "2", "3"]:
                        agmID = sensornames[int(element)]

                    #If element contains actual data (encoded in b64)
                    #decode it and store in the correct IMU(0-6) and the correct sensor(A,G,M)
                    else:
                        reading = getXYZ(element)
                        
                        new_row = pd.DataFrame()
                        new_row = { 
                                    str(agmID + "x"): (reading[x_idx]),
                                    str(agmID + "y"): (reading[y_idx]),
                                    str(agmID + "z"): (reading[z_idx]),
                                }
                        
                        targetdfID = str(imuID + agmID)
                        data[targetdfID] = data[targetdfID].append(new_row, ignore_index=True)

    print("Succesfully Fetched Data from " + fileName)

    #Set to 8 sensors because numbering skips channel 2 (0,1,3,4,5,6,7)
    return joinData(data, 8)


#Put together the A,G,M for each IMU into a df with 9 columns (ax,ay,az, gx,gy,gz, mx,my,mz)
#account for different sampling rates with hold or linear interpolation
def joinData(data: dict, imuCount: int) -> dict[str, pd.DataFrame]:

    #in Hz, samples/second. 
    accgyr_rate = 100
    mag_rate = 100

    joinedData = {}

    #create a dict of dataframes for each imu (9 columns each)
    for idx in range(0, imuCount):

        acc = data["imu" + str(idx) + "_acc"]
        gyr = data["imu" + str(idx) + "_gyr"]
        mag = applyHold(data["imu" + str(idx) + "_mag"], int(accgyr_rate/mag_rate))

        joinedData['imu' + str(idx)] = pd.concat([acc, gyr, mag], axis=1)

    return joinedData

#Transform a timeseries dataframe to match a higher sampling rate by applying a hold of previously sampled value
#factor is the scaling factor (integer 1 <) for time
def applyHold(df: pd.DataFrame, factor: int) -> pd.DataFrame:

    dfnew = pd.DataFrame()
    columns = df.columns.values.tolist()

    new_row = pd.DataFrame()
    for idx, row in df.iterrows():
        
        new_row = { 
                    str(columns[0]): row[0],
                    str(columns[1]): row[1],
                    str(columns[2]): row[2],
                }

        for rep in range(0, factor):     
            dfnew = dfnew.append(new_row, ignore_index=True)

    return dfnew
        

if __name__ == '__main__':
    main()