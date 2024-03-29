from functools import lru_cache
from typing import List
import sys

REGEX_TEXT = ("(?<=[a-z0-9])(?=[A-Z])|"
              "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
              "(?<=[0-9])(?=[a-zA-Z])|"
              "(?<=[A-Za-z])(?=[0-9])|"
              "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
              "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
              "_|\\s+")

if sys.version_info >= (3, 7):
    import re
    SPLIT_REGEX = re.compile(REGEX_TEXT)
else:
    import regex
    SPLIT_REGEX = regex.compile("(?V1)"+REGEX_TEXT)


@lru_cache(maxsize=5000)
def split_identifier_into_parts(identifier: str) -> List[str]:
    identifier_parts = list(s for s in SPLIT_REGEX.split(identifier) if len(s)>0)

    if len(identifier_parts) == 0:
        return [identifier]
    return identifier_parts

if __name__ == '__main__':
    aa = 'public void className <spt> (int AaAbb =1; B_B=0;) .'
    bb = split_identifier_into_parts(aa)
    print(bb)