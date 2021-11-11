import datetime
from pydantic import BaseModel


class BccaasPayload(BaseModel):
    image_base64: str
    datetime: datetime.datetime  # str in ISO 8601 format (YYYY-MM-DDTHH:MM:SS+08:00) e.g. 2021-07-14T15:53:00+08:00
    debug_image: bool  # return individual debug image

