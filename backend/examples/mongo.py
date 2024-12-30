from pymongo import MongoClient


def get_database():
    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    uri = ("mongodb+srv://thienphunhc:ZoyJcHZ0GvjaE3Db@chatbotagent.1hfpp.mongodb.net/")


    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(uri)
    db = client['chatbotagent']
    collection = db['chatbot']
    collection.create_index("SessionId")

    # Create the database for our example (we will use the same database throughout the tutorial
    return client['user_shopping_list']


# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":
    # Get the database
    dbname = get_database()
