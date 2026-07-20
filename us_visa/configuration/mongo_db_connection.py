"""
MongoDB Atlas connection.

The connection string is never hard-coded — it is read from the
MONGODB_URL environment variable (kept in GitHub Secrets on CI and
exported locally), following the same practice used in production.
"""

import os
import sys

import certifi
import pymongo

from us_visa.constants import DATABASE_NAME, MONGODB_URL_KEY
from us_visa.exception import USvisaException
from us_visa.logger import logging

ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable {MONGODB_URL_KEY} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection successful.")
        except Exception as e:
            raise USvisaException(e, sys) from e
