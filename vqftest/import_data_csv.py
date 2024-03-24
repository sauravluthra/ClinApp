import numpy as np
import pandas as pd
import os
from vqf import VQF, offlineVQF
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import remove_frame
from datetime import datetime

#13 and 69
#***********************************************
def main():

#Saurav 
#Bending.csv 
#Bending2.csv 
#SideBending.csv 
#Static.csv 
#Static2.csv 
#Walking_1.csv 
#Walking_2.csv
    
#Lian
#ForwardBending01.csv 
#SideBending01.csv 
#Static.csv 
#Walking_1.csv 
#Walking_2.csv
    
    #ADJUST ALL RUN-TIME PARAMETERS HERE
    params = {
        'CSV_FILENAME'         : "imudata/41XG62/SubjectL/Walking_2.csv", #File Name of csv
        'OUTPUT_DIR'           : "",
        'IMU_COUNT'            : 5, #Number of IMU sensors
        'FIRST_ROW'            : 6, #First row of data (Do NOT change)
        'LAST_ROW'             : 2000, #Last row of data to read
        'SAMPLING_TIME_SENSOR' : 0.005, #real-world sampling period of the sensor in seconds
        'SAMPLING_TIME_VQF'    : 0.01 #sampling time in seconds for the VQF filter
    }

    outputDir = params['CSV_FILENAME']
    outputDir = f"output/{outputDir[outputDir.rfind('/') - 1 : -4]}/"
    params['OUTPUT_DIR'] = outputDir

    if not os.path.isdir(params['OUTPUT_DIR']):
        os.makedirs(params['OUTPUT_DIR'])
    
    print(f"OUTPUT_DIR : {params['OUTPUT_DIR']}")

    #Get each IMU's dataframe
    dfArr = fetchData(params)

    #Run each IMU's dataframe through VQF
    for id in range(params['IMU_COUNT']):
        runvqf(dfArr[id], id, params)

    print("\n\nEXIT PROGRAM")
    return 0

def runvqf(data, imuID, params):

    data = downSampledf(data, (params['SAMPLING_TIME_VQF'] / params['SAMPLING_TIME_SENSOR']))

    #Must convert the dataframe to a np array for VQF to work
    try:
        gyr = np.ascontiguousarray(data[['gx', 'gy', 'gz']].to_numpy())
        acc = np.ascontiguousarray(data[['ax', 'ay', 'az']].to_numpy())
        mag = np.ascontiguousarray(data[['mx', 'my', 'mz']].to_numpy())
    except:
        print("ERROR: CANT CONVERT TO NP CONTIGUOUS ARRAY")
        return

    #Be careful to set this correctly in params
    Ts = params['SAMPLING_TIME_VQF']

    # Run orientation estimation
    vqf9D = offlineVQF(gyr, acc, mag,Ts)

    vqf6D = VQF(Ts)
    out6D = vqf6D.updateBatch(gyr, acc)
    #out_quat = vqf.updateBatch()
    #out_rest = vqf.getRestDetected()
    plt.figure()
    plt.subplot(211)
    plt.plot(out6D['quat6D'])
    plt.title(f"QUATERNION 6D w/BIAS ESTIMATE | IMU#{str(imuID)} | {params['CSV_FILENAME']}")
    plt.grid()


    plt.subplot(212)
    plt.plot(vqf9D['quat9D'])
    plt.grid()
    plt.title(f"QUATERNION 9D w/BIAS ESTIMATE | IMU#{str(imuID)} | {params['CSV_FILENAME']}")
    plt.tight_layout()

    np.set_printoptions(threshold=np.inf)
    now = datetime.now()
    nowStr = now.strftime("%Y%m%d_%H%M%S")

    plt.savefig(os.path.join(params['OUTPUT_DIR'], f'Quat9D_IMU{imuID}_{nowStr}.png'))
    plt.show()

    outputFile6D = os.path.join(params['OUTPUT_DIR'], f'Quat6D_IMU{imuID}_{nowStr}.csv')
    with open(outputFile6D, "w+"):
        np.savetxt(outputFile6D, out6D['quat6D'], delimiter = ",")

    outputFile9D = os.path.join(params['OUTPUT_DIR'], f'Quat9D_IMU{imuID}_{nowStr}.csv')
    with open(outputFile9D, "w+"):
        np.savetxt(outputFile9D, vqf9D['quat9D'], delimiter = ",")

    return 0

#Return a dataframe with every nth row to essentially downsample a high sensor rate
def downSampledf(df, n):
    #return df.iloc[::n, :]
    return df[df.index % n == 0]  # Selects every nth row starting from 0

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

    print(f"FETCH DATA : {params['CSV_FILENAME']}")
    #Reads in the csv file and only stores the relevant rows of data (excl.headers and additional data on bottom)
    csv_df = pd.read_csv(params['CSV_FILENAME'], header=3, on_bad_lines='warn')
    csv_df = csv_df.loc[params['FIRST_ROW'] : params['LAST_ROW']]
    
    #Rename all columns to "col[number]"
    #csv_df = csv_df.rename(columns={'Devices' : 'col_0'})
    #for idx in range(csv_df.shape[1]):
        #csv_df = csv_df.rename(columns={f'Unnamed: {idx}' : f'col_{idx}'})
    
    #Print out for sanity check
    print(csv_df.head())

    #Iterate through every row and move that IMU's 9 DOF readings to the corresponding IMU dataframe
    for idx, row in csv_df.iterrows():

        #This calculates the timestamp in seconds of a sample, given the frame, subfram, and sensor sampling rate
        timestamp = ((int(row['Frame']) * 10) + int(row['Sub Frame'])) * params['SAMPLING_TIME_SENSOR']

        #Iterate through each IMU and pick out its AX,AY,AZ,GX,GY,GZ,MX,MY,MZ readings
        for imuID in range(params['IMU_COUNT']):

            #New row for a given IMU's dataframe
            new_row = pd.DataFrame()
            new_row = { "milliseconds": timestamp,
                       
                        #Divide by 1000.0 to convert mms^-2 => ms^-2
                        "ax": float(row[f"ACCX{imuID + 1}"])/1000.0,
                        "ay": float(row[f"ACCY{imuID + 1}"])/1000.0,
                        "az": float(row[f"ACCZ{imuID + 1}"])/1000.0,

                        #Convert from degrees => radians
                        "gx": np.deg2rad(float(row[f"GYROX{imuID + 1}"])),
                        "gy": np.deg2rad(float(row[f"GYROY{imuID + 1}"])),
                        "gz": np.deg2rad(float(row[f"GYROZ{imuID + 1}"])),

                        "mx": float(row[f"MAGX{imuID + 1}"]),
                        "my": float(row[f"MAGY{imuID + 1}"]),
                        "mz": float(row[f"MAGZ{imuID + 1}"])
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