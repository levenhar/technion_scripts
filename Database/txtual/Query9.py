import psycopg2

try:
    connection = psycopg2.connect(user="postgres",
    password= "mortchelet",
    host = "127.0.0.1",
    port = "5432",
    database = "Big")
    cursorr = connection.cursor()

    update_Query_OwnParcelName = 'update \"Own\" set \"ParcelName\" = LTRIM(RTRIM(\"ParcelName\"))'
    cursorr.execute(update_Query_OwnParcelName)

    postgreSQL_select_Query = 'SELECT count(*) AS num_people_w FROM (SELECT distinct \"ID\" FROM \"Own\" WHERE \"ParcelName\" = \'W\') AS A '

    cursorr.execute(postgreSQL_select_Query)
    num_people_w = cursorr.fetchall()

    print(f"the number of pepole who have parcel named \'W\' is {num_people_w[0][0]}")

except (Exception, psycopg2.DatabaseError) as error:
    print("Error while creating PostgreSQL table", error)

finally:
    cursorr.close()
    connection.close()
