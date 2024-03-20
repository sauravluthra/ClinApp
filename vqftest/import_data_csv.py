import numpy as np
import pandas as pd
from vqf import VQF, offlineVQF
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import remove_frame
from datetime import datetime

#***********************************************
def main():

    #ADJUST ALL RUN-TIME PARAMETERS HERE
    params = {
        'CSV_FILENAME' : "imudata/0103-A-UpAndGo_CROW_trial1.csv", #File Name of csv
        'IMU_COUNT' : 4, #Number of IMU sensors
        'FIRST_ROW': 6, #First row of data (Do NOT change)
        'LAST_ROW': 30000, #Last row of data to read
        'SAMPLING_TIME' : 0.0005 #sampling time in seconds
    }

    #Get each IMU's dataframe
    dfArr = fetchData(params)

    #Run each IMU's dataframe through VQF
    for id in range(params['IMU_COUNT']):
        runvqf(dfArr[id], id, params)

    print("\n\nEXIT PROGRAM")
    return 0

def runvqf(data, imuID, params):

    #Must convert the dataframe to a np array for VQF to work
    try:
        gyr = np.ascontiguousarray(data[['gx', 'gy', 'gz']].to_numpy())
        acc = np.ascontiguousarray(data[['ax', 'ay', 'az']].to_numpy())
        mag = np.ascontiguousarray(data[['mx', 'my', 'mz']].to_numpy())
    except:
        print("ERROR: CANT CONVERT TO NP CONTIGUOUS ARRAY")
        return

    #Be careful to set this correctly in params
    Ts = params['SAMPLING_TIME']

    # Run orientation estimation
    vqf = offlineVQF(gyr, acc, mag,Ts)
    #out_quat = vqf.updateBatch()
    #out_rest = vqf.getRestDetected()

    plt.subplot(212)
    plt.plot(vqf['quat9D'])
    plt.grid()
    plt.title('QUATERNION 9D w/BIAS ESTIMATE | IMU#' + str(imuID))
    plt.tight_layout()
    plt.show()

    np.set_printoptions(threshold=np.inf)
    now = datetime.now()
    nowStr = now.strftime("%Y%m%d_%H%M%S")
    with open(f"output/QuatOutput9D_IMU{imuID}_{nowStr}.csv", "w+"):
        np.savetxt(f"output/QuatOutput9D_IMU{imuID}_{nowStr}.csv", vqf['quat9D'], delimiter = ",")


def fetchData(params):

    # Create a list of dataframes representing each IMU sensor's data
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df6 = pd.DataFrame()
    df7 = pd.DataFrame()
    dfArr = [df1, df2, df3, df4, df5, df6, df7]

    #Reads in the csv file and only stores the relevant rows of data (excl.headers and additional data on bottom)
    csv_df = pd.read_csv(params['CSV_FILENAME'], low_memory=False).loc[params['FIRST_ROW'] : params['LAST_ROW']]
    
    #Rename all columns to "col[number]"
    csv_df = csv_df.rename(columns={'Devices' : 'col_0'})
    for idx in range(csv_df.shape[1]):
        csv_df = csv_df.rename(columns={f'Unnamed: {idx}' : f'col_{idx}'})
    
    #Print out for sanity check
    print(csv_df.head())

    #Iterate through every row and move that IMU's 9 DOF readings to the corresponding IMU dataframe
    for idx, row in csv_df.iterrows():

        #This is calculated as FRAME * 10 + SUBFRAME - Maybe change this?
        timestamp = (int(row['col_0']) * 10) + int(row['col_1'])

        #Iterate through each IMU and pick out its AX,AY,AZ,GX,GY,GZ,MX,MY,MZ readings
        for imuID in range(params['IMU_COUNT']):

            #New row for a given IMU's dataframe
            new_row = pd.DataFrame()
            new_row = { "milliseconds": timestamp,
                       
                        #Divide by 1000.0 to convert mms^-2 => ms^-2
                        "ax": float(row[f"col_{28 + (9 * imuID)}"])/1000.0,
                        "ay": float(row[f"col_{29 + (9 * imuID)}"])/1000.0,
                        "az": float(row[f"col_{30 + (9 * imuID)}"])/1000.0,

                        #Convert from degrees => radians
                        "gx": np.deg2rad(float(row[f"col_{31 + (9 * imuID)}"])),
                        "gy": np.deg2rad(float(row[f"col_{32 + (9 * imuID)}"])),
                        "gz": np.deg2rad(float(row[f"col_{33 + (9 * imuID)}"])),

                        "mx": float(row[f"col_{34 + (9 * imuID)}"]),
                        "my": float(row[f"col_{35 + (9 * imuID)}"]),
                        "mz": float(row[f"col_{36 + (9 * imuID)}"])
                      }
            #Add row to IMU's dataframe
            dfArr[imuID] = dfArr[imuID].append(new_row, ignore_index=True)

    #Success Message
    print(f"Succesfully Fetched Data from {params['CSV_FILENAME']}\n")
    
    #Print for sanity check
    for d in dfArr:
       print(f"datadatadata\n\n{str(d)}")

    return dfArr

if __name__ == '__main__':
    main()