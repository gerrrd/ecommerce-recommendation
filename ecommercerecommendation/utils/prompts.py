"""
This script has a Class implementation to handle prompts for GenAI models.

"""

from __future__ import annotations

from typing import Any, Dict, Union

from ecommercerecommendation.utils.strings import format_str


class Prompt:
    """
    A class to represent prompts and prompt templates to be used in GenAI
    models with a formatting function, too.
    Attribute:
        __prompt__ (str): text of the prompt.

    """

    def __init__(
        self,
        prompt_str: str,
        replace: Union[None, Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Constructs self.__prompt__ removing the possible comments from the
        given input. Parameters are replaced if given as arguments,
        see method: format_str().

        If prompt_str is given, it will be used. If not, prompt path will be
        used. If None of them, them, then an empty string is set.

        :param prompt_str: text of the prompt (template) to be used
        :type prompt_str: Union[str, None]
        :param replace: a dictionary with pairs to be replaced as in
                        format_str() method
        :type replace: Union[None, Dict[str, Any]]
        :param **kwargs: Arbitrary keyword arguments for replacing the keys
                         with the values.
        """
        if isinstance(prompt_str, str):
            self.__prompt__ = prompt_str
            self.remove_comments()
            self.format(replace, **kwargs)
        else:
            self.__prompt__ = ""

    def copy(self) -> Prompt:
        """
        Copies the current text of the prompt from __prompt__ and returns
        a new instance of the same class with the same text.

        :returns: a copy of the current prompt class.
        """
        return Prompt(prompt_str=self.__prompt__)

    def format(
        self, replace: Union[None, Dict[str, Any]] = None, **kwargs
    ) -> Prompt:
        """
        Formats self.__prompt__ as format_str(...) does.

        :param replace: a dictionary with pairs to be replaced as in
                        format_str() method
        :type replace: Union[None, Dict[str, Any]]
        :param **kwargs: Arbitrary keyword arguments for replacing the keys
                         with the values.
        :returns: the formatted string
        """
        self.__prompt__ = format_str(self.__prompt__, replace, **kwargs)
        return self

    def to_string(self) -> str:
        """
        Returns the value of the text prompt (template) to be used.

        :returns: the prompt
        """
        return self.__prompt__

    def remove_comments(self) -> Prompt:
        """
        Removes all lines staring with "#" character, and then all the
        possible empty lines from the beginning of the string.

        :returns: the string without the comments
        """
        self.__prompt__ = "\n".join(
            [
                line
                for line in self.__prompt__.split("\n")
                if not line.startswith("#")
            ]
        )
        while self.__prompt__[0] == "\n":
            self.__prompt__ = self.__prompt__[1:]

        return self
