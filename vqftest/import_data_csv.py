import numpy as np
import pandas as pd
from vqf import VQF, offlineVQF
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import remove_frame

#***********************************************
def main():

    params = {
        'CSV_FILENAME' : "imudata/0103-A-UpAndGo_CROW_trial1.csv",
        'IMU_COUNT' : 4,
        'FIRST_ROW': 6,
        'LAST_ROW': 30
    }

    dfArr = fetchData(params)

    for id in range(params['IMU_COUNT']):
        runvqf(dfArr[id], id)

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


def fetchData(params):

    # Set axes to account for sensor orientation
    x = 2
    y = 0
    z = 1

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
    
    print(csv_df.head())
    #print(csv_df)

    for idx, row in csv_df.iterrows():

        timestamp = (int(row['col_0']) * 10) + int(row['col_1'])

        for imuID in range(params['IMU_COUNT']):

            new_row = pd.DataFrame()
            new_row = { "milliseconds": timestamp,
                       
                        "ax": float(row[f"col_{28 + (9 * imuID)}"])/1000.0,
                        "ay": float(row[f"col_{29 + (9 * imuID)}"])/1000.0,
                        "az": float(row[f"col_{30 + (9 * imuID)}"])/1000.0,

                        "gx": np.deg2rad(float(row[f"col_{31 + (9 * imuID)}"])),
                        "gy": np.deg2rad(float(row[f"col_{32 + (9 * imuID)}"])),
                        "gz": np.deg2rad(float(row[f"col_{33 + (9 * imuID)}"])),

                        "mx": float(row[f"col_{34 + (9 * imuID)}"]),
                        "my": float(row[f"col_{35 + (9 * imuID)}"]),
                        "mz": float(row[f"col_{36 + (9 * imuID)}"])
                      }
            
            dfArr[imuID] = dfArr[imuID].append(new_row, ignore_index=True)


    print(f"Succesfully Fetched Data from {params['CSV_FILENAME']}\n")
    
    for d in dfArr:
       print(f"datadatadata\n\n{str(d)}")

    return dfArr

if __name__ == '__main__':
    main()