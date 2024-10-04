import psycopg2

try:
    connection = psycopg2.connect(user="postgres",
                                  password="mortchelet",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="Big")
    cursorr = connection.cursor()

    postgreSQL_select_Query = 'SELECT \"parcelName\"' \
                              'FROM \"Parcel\", ' \
                              '(SELECT \"polyID\" PLID FROM \"PartOf\",  (SELECT \"pntID\" PNID FROM \"RegisterBy\" ' \
                              'WHERE \"surveyID" =9840000 or \"surveyID" =9840286 or \"surveyID\" =9840275)A  ' \
                              'WHERE \"pnid\" =\"pntID\") B ' \
                              'WHERE \"polyID\"=\"plid\" '

    cursorr.execute(postgreSQL_select_Query)
    num_of_parcel = cursorr.fetchall()

    for row in num_of_parcel:
        print(row[0])


except (Exception, psycopg2.DatabaseError) as error:
    print("Error while creating PostgreSQL table", error)

finally:
    cursorr.close()
    connection.close()
