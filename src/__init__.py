import configparser

config = configparser.ConfigParser()
config.read("config.ini")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import router_list


def set_router(app: FastAPI):
    for information in router_list:
        app.include_router(
            information["router"],
            prefix=information["prefix"],
            responses={404: {"description": "Not found"}},
        )


def set_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )


app = FastAPI()
set_middleware(app)
set_router(app)