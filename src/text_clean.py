import re
from bs4 import BeautifulSoup

def clean_text(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    x = BeautifulSoup(x, "html.parser").get_text()
    x = re.sub(r"\s+", " ", x).strip()
    return x
