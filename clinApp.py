import argparse
import os
from tkinter import *
from tkinter import ttk
import sqlite3
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


  
def getArgs():
    '''
    Gets all command line arguments
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--createPatientDB", help="delete any existing and create a new patient database cfg/")
    args = parser.parse_args()

    return args

class Patient:

    def __init__(self, firstName, lastName, DOB, address, existingCond, height, weight, spineLen, shoulderWid, kyphosisLen, lordosisLen):
                        #(firstName, lastName, DOB, address, existingCond, height, weight, spineLen, shoulderWid, kyphosisLen, lordosisLen)
    
        self.firstName    = firstName
        self.lastName     = lastName
        self.dob          = DOB
        self.address      = address
        self.existingCond = existingCond
        self.height       = height
        self.weight       = weight
        self.spineLen     = spineLen
        self.shoulderWid  = shoulderWid
        self.kyphosisLen  = kyphosisLen
        self.lordosisLen  = lordosisLen

    def printPatient(self):

        print(f"=========== PATIENT INFO ===========\n")
        print(f"Name    : {self.firstName} {self.lastName}")
        print(f"DOB     : {yyyymmddToReadable(self.dob)}")
        print(f"Address : {self.address}")
        print(f"Height  : {self.height} cm")
        print(f"Weight  : {self.weight} kg")
        print(f"=========== Body Measurments =========== ")
        print(f"Spine Length    : {self.spineLen} cm ")
        print(f"Shoulder Width  : {self.shoulderWid} cm") 
        print(f"Kyphosis Length : {self.kyphosisLen} cm")
        print(f"Lordosis Length : {self.lordosisLen} cm")

    def fetchPatientInfo(self, listOfFields):
        '''
        listOfFields : list containing [firstName, lastName, DOB]
        patient      : the patient object to be populated with the loaded information

        Pulls the patient's personal data from the patient database to populate the clinician's GUI.
        '''

        firstName = listOfFields[0]
        lastName  = listOfFields[1]
        dob       = listOfFields[2]

        conn = sqlite3.connect('cfg/patientInfo.db')
        cursor= conn.cursor()

        loadData = f""" 
                SELECT *
                FROM PATIENTS
                WHERE firstName={firstName} AND lastName={lastName} AND DOB={dob}
                ); """
        
        print("LOADDATA: " + loadData)
        
        res = cursor.execute(loadData)
        conn.close()

        if res != None:
            print("Patient successfully found")
            self.__init__(res[0], res[1],res[2],res[3],res[4],res[5],res[6],res[7],res[8],res[9],res[10])
        else:
            print("Error: Could not find patient in the patient database")
            return -1

    def loadPatientInfo(self):

        addWindow = Tk()

        addWindow.title("Clinician Application - Load Patient Info")
        addWindow.geometry('500x500')

        lbl1 = Label(addWindow, text= 'Search for a patient:')
    
        lbl2 = Label(addWindow, text= 'First Name')
        firstName = Text(addWindow, height = 1,width = 25,bg = "black")

        lbl3 = Label(addWindow, text= 'Last Name')
        lastName = Text(addWindow, height = 1,width = 25,bg = "black")

        lbl4 = Label(addWindow, text= 'Date of Birth (YYYYMMDD)')
        dob = Text(addWindow, height = 1,width = 25,bg = "black")
        
        listOfLabels = [lbl2,      lbl3,     lbl4]
        listOfFields = [firstName, lastName, dob]
        
        enterButton = Button(addWindow, height = 2,
                        width = 20,
                        text ="Search for and Load Patient",
                        command = lambda:self.fetchPatientInfo(self, listOfFields))
        
        lbl1.pack()
        for idx, lab in enumerate(listOfLabels):
            lab.pack()
            listOfFields[idx].pack()
        enterButton.pack()


def yyyymmddToReadable(date):

    months = ['January', 'Feburary', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    if len(date) != 8:
        print("yyyymmddToReadable Error - must pass strings of length = 8")
        return 'Feburary 30, 2024'
    else:
        return str(months[int(date[4:5]) - 1]) + date[6:7] + ', ' + date[: 3]

def createPatientDB():
    '''
    Deletes any existing Patient database and creates a new one with an empty table
    '''

    try:
        os.remove('cfg/patientInfo.db')
        print("Previous version of patientInfo.db removed")
    except:
        print("No existing patientInfo.db was found")

    conn = sqlite3.connect('cfg/patientInfo.db')
    cursor= conn.cursor()
    
    createTable = """ 
                CREATE TABLE PATIENTS (
                firstName     CHAR(50) NOT NULL,
                lastName      CHAR(50) NOT NULL,
                DOB           CHAR(8) NOT NULL,
                address       CHAR(200) NOT NULL,
                existingCond  CHAR(200) NOT NULL,
                height        FLOAT NOT NULL,
                weight        FLOAT NOT NULL,
                spineLen      FLOAT NOT NULL,
                shoulderWid   FLOAT NOT NULL,
                kyphosisLen   FLOAT NOT NULL,
                lordosisLen   FLOAT NOT NULL
                ); """
    
    cursor.execute(createTable)
    print("New patientInfo.db created")

def loadData():
    '''
    Attempts to load the raw data from the connected Patient device and save it into a local file
    '''

    print("ATTEMPTING TO LOAD DATA")

def procData():
    '''
    Runs the patient's raw data through specified filters and processing steps and saves in a new file
    Possibly implemented as a seperate C++ program.
    '''
    print("RUNNING DATA THROUGH FILTERS AND GENERATING A FILE CONTAINING NEW FINDINGS AND FILTERED DATA")

def takeInput(listOfFields):
    '''
    listOfFields : a list of all the fields containted in the patient info database

    Reads the input from the add new patient window and stores it in the patient info database
    '''
    
    input = []

    for field in listOfFields:
        input.append(field.get("1.0", "end-1c"))

    print("Inputted Patient Info: " + str(input))

    conn = sqlite3.connect('cfg/patientInfo.db')
    cursor= conn.cursor()

    addPatient = f""" 
                INSERT INTO PATIENTS VALUES ({input[0]}, {input[1]}, {input[2]}, {input[3]}, {input[4]}, {input[5]}, {input[6]}, {input[7]}, {input[8]}, {input[9]}, {input[10]}); 
                """
    #(firstName, lastName, DOB, address, existingCond, height, weight, spineLen, shoulderWid, kyphosisLen, lordosisLen)
    try:
        cursor.execute(addPatient)
        conn.commit()
        print("Patient successfully added to the patient info database")
    except Exception as e:
        # By this way we can know about the type of error occurring
        print("The error is: ",e)
        print("Error: Could not add new patient")

        
    conn.close()

def addPatientInfo():
    '''
    Open a new window with fields to fill in patient information and store it in the patient info database
    '''

    addWindow = Tk()

    addWindow.title("Clinician Application - Add New Patient")
    addWindow.geometry('500x800')

    lbl1 = Label(addWindow, text= 'Fill this form with accurate patient info (Name, Age, Address, Health Conditions, etc...)')
   
    lbl2 = Label(addWindow, text= 'First Name')
    firstName = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl3 = Label(addWindow, text= 'Last Name')
    lastName = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl4 = Label(addWindow, text= 'Date of Birth (YYYYMMDD)')
    dob = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl5 = Label(addWindow, text= 'Address')
    address = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl6 = Label(addWindow, text= 'Existing Health Conditions')
    existingCond = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl7 = Label(addWindow, text= 'Height (cm)')
    height = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl8 = Label(addWindow, text= 'Weight (kg)')
    weight = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl9 = Label(addWindow, text= 'Spine Length (cm)')
    spineLen = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl10 = Label(addWindow, text= 'Shoulder Width (cm)')
    shoulderWid = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl11 = Label(addWindow, text= 'Kyphosis Length (cm)')
    kyphosisLen = Text(addWindow, height = 1,width = 25,bg = "black")

    lbl12 = Label(addWindow, text= 'Lordosis Length (cm)')
    lordosisLen = Text(addWindow, height = 1,width = 25,bg = "black")
    
    listOfLabels = [lbl2,      lbl3,     lbl4,lbl5,    lbl6,         lbl7,   lbl8,   lbl9,     lbl10,       lbl11,       lbl12]
    listOfFields = [firstName, lastName, dob, address, existingCond, height, weight, spineLen, shoulderWid, kyphosisLen, lordosisLen]
    
    enterButton = Button(addWindow, height = 2,
                    width = 20,
                    text ="Confirm and Enter Info",
                    command = lambda:takeInput(listOfFields))
    
    print("ENTER BUTTON: " + str(enterButton))
    
    lbl1.pack()
    for idx, lab in enumerate(listOfLabels):
        lab.pack()
        listOfFields[idx].pack()
    enterButton.pack()

def main():

    args = getArgs()

    if args.createPatientDB:
        createPatientDB()

    thePatient = Patient('', '', '', '','', 0, 0, 0, 0, 0, 0)
    #(firstName, lastName, DOB, address, existingCond, height, weight, spineLen, shoulderWid, kyphosisLen, lordosisLen)
    
    window = Tk()

    window.title("Clinician Application - Posture Monitoring Device")
    window.geometry('1000x1000')

    tab_control = ttk.Notebook(window)

    tab1 = ttk.Frame(tab_control)
    tab2 = ttk.Frame(tab_control)
    tab3 = ttk.Frame(tab_control)

    tab_control.add(tab1, text='Patient Info')
    tab_control.add(tab2, text='Findings')
    tab_control.add(tab3, text='Data Recording')

    lbl1 = Label(tab1, text= 'This will contain patient information for the clinician to refer to. (Name, Age, Address, Health Conditions, etc...)')
    lbl1.grid(column=0, row=0)
    loadPatient_button = Button(tab1, text ="Load Patient Info", command = thePatient.loadPatientInfo)
    loadPatient_button.grid(column=0, row=1)
    addPatient_button = Button(tab1, text ="Add New Patient", command = addPatientInfo)
    addPatient_button.grid(column=0, row=2)

    lbl2 = Label(tab2, text= 'This will contain generated graphs, key metrics, and other visualizations that may help the clinician interpret the data and make a diagnosis.')
    lbl2.grid(column=0, row=0)

    lbl3 = Label(tab3, text= 'This will contain a button to attempt to load the data from an attached Device (computer USB port), and display data in some ways.')
    lbl3.grid(column=0, row=0)
    loadData_button = Button(tab3, text ="Load Data", command = loadData)
    loadData_button.grid(column=0, row=1)

    tab_control.pack(expand=1, fill='both')

    window.mainloop()

main()
print("\nExited Program\n")