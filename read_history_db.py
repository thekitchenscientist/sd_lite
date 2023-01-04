import random
import config
import copy
import os
import uuid
import hashlib
import sqlite3
from sqlite3 import Error

loaded_pipe = config.loaded_pipe

BUFFER_SIZE = 65536 
# ensure correct folders exist and are used
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))

# assign unique ID's to each configuration chosen
def check_config_hash(filepath):

    md5 = hashlib.md5()
    with open(filepath,'rb') as config_file:
        while True:
            data = config_file.read(BUFFER_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()

SESSION_ID = check_config_hash(ROOT_DIR+"/config.py")

CONFIG_LIST = (SESSION_ID,config.SPLIT_ATTENTION,config.MEMORY_EFFICIENT_ATTENTION,config.HALF_PRECISION,config.MODEL_ID,config.IMAGE_INPUT_FOLDER,config.IMAGE_OUTPUT_FOLDER,config.IMAGE_FORMAT,config.IMAGE_SCHEDULER,
                   config.IMAGE_WIDTH,config.IMAGE_HEIGHT,config.IMAGE_SEED,config.IMAGE_SCALE,config.IMAGE_STEPS,config.IMAGE_SCALE_OFFSET,config.IMAGE_STEPS_OFFSET,config.IMAGE_COUNT,config.IMAGE_STRENGTH,config.IMAGE_STRENGTH_OFFSET,
                   config.IMAGE_BRACKETING,config.SAVE_METADATA_TO_IMAGE)

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        #print(sqlite3.version)
    except Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def create_history_database():
    database = ROOT_DIR+"/history.db"

    sql_create_config_table = """ CREATE TABLE IF NOT EXISTS config (
                                        hash text PRIMARY KEY,
                                        SPLIT_ATTENTION integer NOT NULL,
                                        MEMORY_EFFICIENT_ATTENTION  integer NOT NULL,
                                        HALF_PRECISION integer NOT NULL,
                                        MODEL_ID text NOT NULL,
                                        IMAGE_INPUT_FOLDER text,
                                        IMAGE_OUTPUT_FOLDER text,
                                        IMAGE_FORMAT text NOT NULL,
                                        IMAGE_SCHEDULER text NOT NULL,
                                        IMAGE_WIDTH integer NOT NULL,
                                        IMAGE_HEIGHT integer NOT NULL,
                                        IMAGE_SEED integer NOT NULL,
                                        IMAGE_SCALE real NOT NULL,
                                        IMAGE_STEPS integer NOT NULL,
                                        IMAGE_SCALE_OFFSET real NOT NULL,
                                        IMAGE_STEPS_OFFSET integer NOT NULL,
                                        IMAGE_COUNT integer NOT NULL,
                                        IMAGE_STRENGTH real NOT NULL,
                                        IMAGE_STRENGTH_OFFSET real NOT NULL,
                                        IMAGE_BRACKETING integer NOT NULL,
                                        SAVE_METADATA_TO_IMAGE integer NOT NULL
                                    ); """

    sql_create_prompt_table = """CREATE TABLE IF NOT EXISTS prompts (
                                    id integer PRIMARY KEY,
                                    config_hash text NOT NULL,
                                    UUID text NOT NULL,
                                    scheduler text NOT NULL,
                                    prompt text,
                                    anti_prompt text,
                                    steps  integer NOT NULL,
                                    SCALE real NOT NULL,
                                    seed integer NOT NULL,
                                    n_images integer NOT NULL,
                                    date_time text NOT NULL,
                                    FOREIGN KEY (config_hash) REFERENCES config (hash)
                                );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_config_table)
        # create tasks table
        create_table(conn, sql_create_prompt_table)
    else:
        print("Error! cannot create the database connection.")


def add_config_hash(conn, config_list):
    """
    Create a entry in the config table
    :param conn:
    :param CONFIG_LIST:
    :return: row id
    """
    sql = ''' INSERT INTO config(hash, 
                                        SPLIT_ATTENTION,
                                        MEMORY_EFFICIENT_ATTENTION,
                                        HALF_PRECISION,
                                        MODEL_ID,
                                        IMAGE_INPUT_FOLDER,
                                        IMAGE_OUTPUT_FOLDER,
                                        IMAGE_FORMAT,
                                        IMAGE_SCHEDULER,
                                        IMAGE_WIDTH,
                                        IMAGE_HEIGHT,
                                        IMAGE_SEED,
                                        IMAGE_SCALE,
                                        IMAGE_STEPS,
                                        IMAGE_SCALE_OFFSET,
                                        IMAGE_STEPS_OFFSET,
                                        IMAGE_COUNT,
                                        IMAGE_STRENGTH,
                                        IMAGE_STRENGTH_OFFSET,
                                        IMAGE_BRACKETING,
                                        SAVE_METADATA_TO_IMAGE)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, config_list)
    conn.commit()
    return cur.lastrowid

def check_config_recorded(conn, config_hash):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM config WHERE hash=?", (config_hash,))

    rows = cur.fetchall()

    return rows

def dump_prompts_to_console(conn):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM prompts")

    rows = cur.fetchall()

    for row in rows:
        print(row)

def dump_config_to_console(conn):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM config")

    rows = cur.fetchall()

    for row in rows:
        print(row)


#if it exists dump contents
if os.path.exists(ROOT_DIR+"/history.db"):
    conn = create_connection(ROOT_DIR+"/history.db")
    dump_config_to_console(conn)
    dump_prompts_to_console(conn)
else:
    print("history.db not found")