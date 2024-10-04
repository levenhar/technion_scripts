from tkinter import *
from tkinter import messagebox
import psycopg2 as pg
import time
from tkinter import ttk



    
class UI:
    #class that save all the information needed to the GUI
    def __init__(self, root):
        self.root = root  #the main window of the GUI
        self.Fname="" #first name for query definition. adjusted to LIKE operator in SQL
        self.Lname="" #last name for query definition. adjusted to LIKE operator in SQL
        self.ID= "" #ID for query definition. adjusted to LIKE operator in SQL
        self.user="" #user for database connection
        self.password="" #password for database connection
        self.host="" #host for database connection
        self.port="" #port for database connection
        self.database="" #database name for database connection
        '''the attribute below is a aids filed for saving the input information'''
        self.Fname_1 = StringVar()
        self.Lname_1 = StringVar()
        self.ID_1 = StringVar()
        self.user_1=StringVar()
        self.password_1=StringVar()
        self.host_1=StringVar()
        self.port_1=StringVar()
        self.database_1=StringVar()
        ''''''
        self.isconnected = False #indecate if the connection to the database succeeded
        self.conn = None #save the connection to the database
        self.curs = None
        self.person = None #save the update person list that present on the GUI
        self.Parcel = None #save the update Parcel list that present on the GUI
        self.person_Table = None #the object that disply on the GUI as person table
        self.parcel_Table = None #the object that disply on the GUI as parcel table
        self.IDforQuery = "" #string that defined the percal query, given by click on the relevent person in person table.
        messagebox.showinfo("DataBase", "Please enter a DataBase information")



    def Quit(self):  # Close the program
        self.root.quit()
        self.root.destroy()
        self.curs.close()
        self.conn.close()



    def Search(self):
        '''
        function that take the input query, select the relevant person from the database and present them on the GUI.
        all the input type will be string and will be adjusted to LIKE opertor in SQL
        '''
        self.Fname = (self.Fname_1.get())
        self.Lname = (self.Lname_1.get())
        self.ID = (self.ID_1.get())
        if not self.isconnected:
            messagebox.showinfo("Error", "you need to connect to the Database first")

        #build the query statement
        elif self.Fname or self.Lname or self.ID:
            Query = 'SELECT \"ID\", \"firstName\", \"lastNAme\", \"address\" FROM \"Person\" Where '
            if self.Fname :
                Query += f'\"firstName\" LIKE \'{self.Fname}\''
                if self.Lname:
                    Query += f'and \"lastNAme\" LIKE \'{self.Lname}\''
                if self.ID:
                    Query += f' and \"ID\" LIKE \'{self.ID}\''
            elif self.Lname :
                Query += f'\"lastNAme\" LIKE \'{self.Lname}\''
                if self.ID:
                    Query += f' and \"ID\" LIKE \'{self.ID}\''
            elif self.ID :
                Query += f'\"ID\" LIKE \'{self.ID}\''
            start_time = time.time() #save the specific time
            self.curs.execute(Query) #send the query to postgresql
            self.person = self.curs.fetchall()
            end_time = time.time() #save the specific time

            #deleting the current table
            for i in self.person_Table.get_children():
                self.person_Table.delete(i)

            #insert the new values acordding to the query
            for P in self.person:
                self.person_Table.insert("", "end", values=P)


            amount = len(self.person) #calculat the number of lines
            timee = round(end_time-start_time,5) #calculate searching time
            messagebox.showinfo("searching time", f"amount of results - {amount} \n searching time [sec] - {timee}")
        else:
            messagebox.showinfo("Error", "No query data defined")


    def selectItem(self,a):
        '''
        function that will start after the user will click on one person in the person table.
        the function find his parcels and display it in the parcel table.

        '''
        curItem = self.person_Table.focus()
        self.IDforQuery = self.person_Table.item(curItem).get('values')[0]

        ParcelQuery = f'SELECT \"pn\", \"bl\",\"plid\", \"municipalAuthority\", \"surveyID\", \"area\" ' \
                      'FROM \"Polygon\",' \
                      '(SELECT \"pn\", \"bl\",\"plid\", \"municipalAuthority\", \"polyID\" PY, \"surveyID\" ' \
                      'FROM \"MutationPlan\",' \
                      '(SELECT \"pn\", \"bl\", \"planID\" PLID, \"municipalAuthority\", \"polyID\" ' \
                      'FROM \"Block\", ' \
                      f'(SELECT \"ParcelName\" PN, \"BlockID\" BL FROM \"Own\" WHERE \"ID\" = {self.IDforQuery}) A ' \
                      'WHERE \"bl\" = \"blockID\") B ' \
                      'WHERE \"planID\" = \"plid\") C ' \
                      'WHERE \"py\" = \"polyID\"'
        start_time1 = time.time()
        self.curs.execute(ParcelQuery)
        self.Parcel = self.curs.fetchall()
        end_time1 = time.time()

        # deleting the current table, goes on all the lines in parcel table
        for i in self.parcel_Table.get_children():
            self.parcel_Table.delete(i)

        # insert the new values acordding to the query, goes on all the lines in parcel list.
        for P in self.Parcel:
            self.parcel_Table.insert("", "end", values=P)

        totalArea = 0
        #loop that calculate the total area, its goes on parcel list and sumarize the last element which is the area.
        for row in self.Parcel:
            totalArea += row[-1]

        timee1 = round(end_time1 - start_time1, 5)
        messagebox.showinfo("searching time", f"Total area [m] - {totalArea} \n searching time [sec] - {timee1}")


    def ui(self):
        # Place the Buttons on the screen
        Button(text="Search", command=self.Search).grid(row=3, column=7)
        Button(text="Quit", command=self.Quit).grid(row=0, column=10)

        # Place the Titles on the screen
        Label(self.root, text="DataBase Information", font=('Times New Roman', 15)).grid(row=0, column=0)
        Label(self.root, text="Query Attributes", font=('Times New Roman', 15)).grid(row=2, column=0)
        Label(self.root, text="Person Table", font=('Times New Roman', 15)).grid(row=4, column=0)
        Label(self.root, text="Parcel Table", font=('Times New Roman', 15)).grid(row=6, column=0)

        # make a place for input query information
        Label(self.root, text="First Name",font=('Times New Roman', 10)).grid(row=3, column=0)
        Label(self.root, text="Last Name", font=('Times New Roman', 10)).grid(row=3, column=2)
        Label(self.root, text="ID", font=('Times New Roman', 10)).grid(row=3, column=4)
        Entry(self.root, textvariable=self.Fname_1).grid(row=3, column=1)
        Entry(self.root, textvariable=self.Lname_1).grid(row=3, column=3)
        Entry(self.root, textvariable=self.ID_1).grid(row=3, column=5)

        # make a place for input database information
        Label(self.root, text="User", font=('Times New Roman', 10)).grid(row=1, column=0)
        Label(self.root, text="Password", font=('Times New Roman', 10)).grid(row=1, column=2)
        Label(self.root, text="Host", font=('Times New Roman', 10)).grid(row=1, column=4)
        Label(self.root, text="Port", font=('Times New Roman', 10)).grid(row=1, column=6)
        Label(self.root, text="DataBase Name", font=('Times New Roman', 10)).grid(row=1, column=8)
        Entry(self.root, textvariable=self.user_1).grid(row=1, column=1)
        Entry(self.root, textvariable=self.password_1).grid(row=1, column=3)
        Entry(self.root, textvariable=self.host_1).grid(row=1, column=5)
        Entry(self.root, textvariable=self.port_1).grid(row=1, column=7)
        Entry(self.root, textvariable=self.database_1).grid(row=1, column=9)
        Button(self.root, text="Apply DB", command=self.ApplyLoading).grid(row=1, column=10)



    def ApplyLoading(self):
        '''
        function that take the input database information and try to connect with the database.
        if it succeed, the function update the firstName column in person table and removing all the spaces there.
        after that it display the two table.

        '''
        self.user = (self.user_1.get())
        self.password = (self.password_1.get())
        self.host = (self.host_1.get())
        self.port = (self.port_1.get())
        self.database = (self.database_1.get())
        try:
            self.conn = self.conn = pg.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database)

            self.isconnected=True
            self.curs = self.conn.cursor()

            #deleting spaces in person first name
            update_Query_PrsonFname = 'update \"Person\" set \"firstName\" = LTRIM(RTRIM(\"firstName\"))'
            self.curs.execute(update_Query_PrsonFname)

            # change variable type from int to str
            update_Query_PrsonID = 'ALTER TABLE \"Person\" ALTER COLUMN \"ID\" TYPE VARCHAR(15);'
            self.curs.execute(update_Query_PrsonID)


            personQuery = 'SELECT \"ID\", \"firstName\", \"lastNAme\", \"address\" FROM \"Person\"'
            self.curs.execute(personQuery)
            self.person = self.curs.fetchall()

            #define and display person table
            cols1 = ["ID","First Name", "Last Name", "Address"]
            self.person_Table = ttk.Treeview(self.root,columns = cols1, show="headings")
            for col in cols1:
                self.person_Table.heading(col, text = col)
            self.person_Table.grid(row=5, column=0, columnspan=10)
            for P in self.person:
                self.person_Table.insert("", "end", values=P)

            # define and display parcel table
            cols2 = ["Parcel Name", "BlockID", "Mutation Plan","Municipal Auth", "surveyID", "Area"]
            self.parcel_Table = ttk.Treeview(self.root, columns=cols2, show="headings")
            for col in cols2:
                self.parcel_Table.heading(col, text=col)
            self.parcel_Table.grid(row=7, column=0, columnspan=10)

            #define click event
            self.person_Table.bind('<ButtonRelease-1>', self.selectItem)

            messagebox.showinfo("loaded successfully", "The database has loaded successfully")




        except pg.DatabaseError:
            messagebox.showinfo("Error",
                                "The database has not loaded \n \n Check your query input")
            print ('I am unable to connect the database')







