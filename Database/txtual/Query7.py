
import psycopg2

try:
    connection = psycopg2.connect(user="postgres",
                                  password="mortchelet",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="Big")
    cursorr = connection.cursor()

    postgreSQL_select_Query = 'SELECT \"plid\" , \"drawDate\",\"registerDate\", \"approvedBy\", \"surveyID\", \"blockID\" ' \
                              'FROM \"MutationPlan\",' \
                              '(SELECT \"blockID\", \"planID\" PLID ' \
                              'FROM \"Block\")A ' \
                              'WHERE \"plid\" = \"planID\" and \"plid\" > 11110899'

    cursorr.execute(postgreSQL_select_Query)
    block_information = cursorr.fetchall()

    for row in block_information:
        print("PlanId = ", row[0], )
        print("DrawDate = ", row[1])
        print("RegistareDate  = ", row[2], )
        print("ApprovedBy = ", row[3], )
        print("SurveyID = ", row[4], )
        print("BlockID = ", row[5], "\n")
except (Exception, psycopg2.DatabaseError) as error:
    print("Error while creating PostgreSQL table", error)

finally:
    cursorr.close()
    connection.close()
