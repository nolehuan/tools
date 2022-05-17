import sqlite3

'''
sqlite3存在系统表sqlite_master,结构如下：
sqlite_master(
    type TEXT,      #类型:table-表,index-索引,view-视图
    name TEXT,      #名称:表名,索引名,视图名
    tbl_name TEXT,
    rootpage INTEGER,
    sql TEXT
    )
'''
# 查看某数据库中所有表
def GetTables(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        # cur.execute("select name from sqlite_master where type='table' order by name")
        cur.execute("select data from keypoints limit 1")
        print(cur.fetchall())
        cur.close()
        conn.close()
    except sqlite3.Error as e:
            print(e)

# GetTables("../dataset/SFM/kitti/kitti_feature.db")

def Tofile(data, filename):
    with open(filename, 'wb') as file:
        file.write(data)

def readDB(imgId, db_file):
    try:
        sqliteConnection = sqlite3.connect(db_file)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sql_fetch_blob_query = """SELECT * from colors where image_id = ?"""
        cursor.execute(sql_fetch_blob_query, (imgId,))
        record = cursor.fetchall()
        for row in record:
            print("Id = ", row[0])
            id = row[0]
            data = row[3]

            print("Storing image on disk")
            Path = "./" + str(id) + ".txt"
            Tofile(data, Path)

        cursor.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection is closed")

readDB(0, "../dataset/SFM/kitti/kitti_feature.db")

# https://blog.csdn.net/lsllll44/article/details/116573316
def writeTofile(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)
    print("Stored blob data into: ", filename)

def readBlobData(imgId, db_file):
    try:
        sqliteConnection = sqlite3.connect(db_file)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sql_fetch_blob_query = """SELECT * from images where imgId = ?"""
        cursor.execute(sql_fetch_blob_query, (imgId,))
        record = cursor.fetchall()
        for row in record:
            print("Id = ", row[0])
            id = row[0]
            img = row[1]

            print("Storing image on disk")
            imgPath = "./" + str(id) + ".jpg"
            writeTofile(img, imgPath)

        cursor.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("sqlite connection is closed")

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData

def insertBLOB(imgId, img_file, db_file):
    try:
        sqliteConnection = sqlite3.connect(db_file)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        sqlite_insert_blob_query = """INSERT INTO images (imgId, img) VALUES (?, ?)"""

        img = convertToBinaryData(img_file)

        data_tuple = (imgId, img)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")

# db_file = "./test.db"
# insertBLOB(0, "./sqlite3test.jpg", db_file)
# readBlobData(0, db_file)
