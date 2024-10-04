import psycopg2

try:
    connection = psycopg2.connect(user="postgres",
                                  password="mortchelet",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="Big")
    cursorr = connection.cursor()

    postgreSQL_select_Query = 'SELECT DISTINCT \"pntID\", \"bparcel\"' \
                              'FROM \"PartOf\",' \
                              '(' \
                              'SELECT \"bparcel\", \"polyID\" CPOLYID' \
                              ' FROM \"Parcel\",' \
                              '(' \
                              'SELECT \"ParcelName\" BPARCEL, \"BlockID\" BBLOCKID' \
                              ' FROM \"Own\",' \
                              '(' \
                              'SELECT \"ID\" IDPERSON' \
                              ' FROM \"Person\"' \
                              ' WHERE \"firstName\"= \'אברהים\' and \"lastNAme\"=\'סרסור\'' \
                               ') A' \
                              ' WHERE \"idperson\"=\"ID\"' \
                               ') B' \
                              ' WHERE \"bparcel\"=\"parcelName\" and \"bblockid\"=\"blockID\"' \
                               ') C' \
                              ' WHERE \"cpolyid\"=\"polyID\"'

    cursorr.execute(postgreSQL_select_Query)
    num_of_parcel_sarsur = cursorr.fetchall()

    for row in num_of_parcel_sarsur:
        print(row[0])
        print(row[1])

except (Exception, psycopg2.DatabaseError) as error:
    print("Error while creating PostgreSQL table", error)

finally:
    cursorr.close()
    connection.close()
