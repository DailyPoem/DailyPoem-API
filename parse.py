import collections
from pyunsplash import PyUnsplash
import requests

pu = PyUnsplash(api_key="2Ji-UWqNuEiv26i_XPtE3jS712RllXQ__n9rXOXbSBw")



for i in range(3, 30):
    photos = pu.photos(type_="random", count=1, featured=True, collections="OjKbQySOz6Q")
    [photo] = photos.entries
    response = requests.get(photo.link_download, allow_redirects=True)
    open(f"./static/image/unsplash_{i}.png", "wb").write(response.content)
