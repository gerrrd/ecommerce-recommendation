from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from ecommercerecommendation.utils.constants import BULLET_POINT, REMOVE_CHARS


def bullet_points(list_str: List[str]) -> str:
    """
    Creates a string with bullet points out of a list of strings.

    :param list_str: separate bullet points
    :return: as a string, concatenated
    """
    return BULLET_POINT + BULLET_POINT.join(list_str)


def remove_chars(s: str) -> str:
    """
    Removes all non-alphanumeric characters, that we want to check about,
    from the string.

    :param s: description
    :return: text without those characters
    """
    s_temp = s
    for c in REMOVE_CHARS:
        s_temp = s_temp.replace(c, "")
    return s_temp


def clean_entry(text: str) -> str:
    """
    Text cleaning functions as presented in the 2nd notebook.

    :param text: description
    :return: the clean description
    """

    return " ".join(
        re.sub(
            r"([0123456789].\")",
            r"\1 ",
            re.sub(
                r"[!()/,]",
                " ",
                text.strip(".")
                .replace(". ", " ")
                .replace("&", " AND ")
                .replace("/", " ")
                .replace(" - ", " "),
            ),
        )
        .strip(" ")
        .split()
    )


def format_str(
    s, replace: Union[None, Dict[str, Any]] = None, **kwargs
) -> str:
    """
    Formats the string as str.format(...) does: replacing {variables} with
    their values. There are two possible inputs:
    The argument "replace" as
        {
            "name": "James",
            "city": "Berlin",
        }
    will replace all {name} to James and {city} to Berlin, or as extra
    arguments given in **kwargs:
        prompt = Prompt(path_to_prompt, name="James", city="Berlin").

    :param s: the text of string to be worked on
    :type s: str
    :param replace: a dictionary with pairs to be replaced, such as:
        {
            "name": "James",
            "city": "Berlin",
        }
    :type replace: Union[None, Dict[str, Any]]
    :param **kwargs: Arbitrary keyword arguments for replacing the keys with
                     the values.
    :returns: the formatted string
    """
    if isinstance(replace, dict):
        replace.update(kwargs)
    else:
        replace = kwargs

    if isinstance(replace, dict):
        for key, value in replace.items():
            s = s.replace("{" + key + "}", str(value))
    return s
