# -*- coding: utf-8 -*-
# @file utils.py
# @author zhangshilong
# @date 2024/7/6

import json
import os
import re
import shutil
import sqlite3
from functools import cache
from typing import Any
from typing import List
from typing import Optional

import pandas as pd


class File:
    @classmethod
    def exists(cls, path: str) -> bool:
        """
        如果path是软链接，exists判断软链接指向的目标是否存在。如果想判断软链接本身是否存在，请用islink
        """
        return os.path.exists(path)

    @classmethod
    def isfile(cls, *args, **kwargs) -> bool:
        return os.path.isfile(*args, **kwargs)

    @classmethod
    def isdir(cls, *args, **kwargs) -> bool:
        return os.path.isdir(*args, **kwargs)

    @classmethod
    def islink(cls, *args, **kwargs) -> bool:
        return os.path.islink(*args, **kwargs)

    @classmethod
    def normalize_path(cls, path: str) -> str:
        """windows 前缀会带有像"D:"的盘名称、用于连接的斜杠与linux不一致"""
        return path.split(":", 1)[-1].replace("\\", "/")

    @classmethod
    def join(cls, *paths: Any) -> str:
        paths = [str(path) for path in paths]
        return cls.normalize_path(os.path.join(*paths))

    @classmethod
    def abspath(cls, path: str) -> str:
        """
        获取path的绝对路径
        :param path: 文件或文件夹路径，如"file", __file__
        :return:
        """
        return cls.normalize_path(os.path.abspath(path))

    @classmethod
    def dirname(cls, path: str) -> str:
        """
        获取path的上一层文件夹路径
        :param path: 文件或文件夹路径，如"file", __file__
        :return:
        """
        return os.path.dirname(cls.abspath(path))

    @classmethod
    def basename(cls, path: str) -> str:
        """
        获取path的最后一层的文件/文件夹名
        :param path: 文件或文件夹路径，如"file", __file__
        :return:
        """
        return os.path.basename(cls.abspath(path))

    @classmethod
    def listdir(cls, path: str) -> List[str]:
        return os.listdir(path)

    @classmethod
    def makedirs(cls, directory: str, exist_ok: bool = True) -> None:
        """
        递归创建目录
        1) 如果directory任意一级存在且不为文件夹，无论exist_ok是什么，都会引发FileExistsError或FileNotFoundError
        2) 如果directory存在且为文件夹，exist_ok=False会引发FileExistsError，exist_ok=True则静默
        3) 如果directory不存在，则能成功创建
        """
        os.makedirs(directory, exist_ok=exist_ok)

    @classmethod
    def remove(cls, path: str, recursive: bool = False) -> None:
        """
        删除目标（文件、文件夹、软链接）；如果recursive=True，则递归向上删除空文件夹
        """
        if cls.islink(path):
            os.remove(path)  # 不会删掉软链接指向的目标
        elif cls.exists(path):
            if cls.isfile(path):
                os.remove(path)
            elif cls.isdir(path):
                shutil.rmtree(path)
            else:
                raise TypeError(f"{path=} exists, but is neither file nor directory")

        if recursive:
            path = cls.dirname(path)
            if not cls.exists(path) or not cls.listdir(path):
                cls.remove(path, True)

    @classmethod
    def copy(cls, src: str, dst: str, cover: bool = False) -> str:
        if cls.isfile(src):
            if cls.isdir(dst):
                dst = cls.join(dst, cls.basename(src))  # 转化为文件字符串，方便判断目标文件是否已存在，并且不改变原copy的返回值
            cls.makedirs(cls.dirname(dst))
            if cls.isfile(dst) and not cover:  # 原copy会默认覆盖原dst文件
                raise FileExistsError(f"file {dst=} already exists")
            return shutil.copy(src, dst)  # 如果dst是存在的文件夹，则返回f"{dst}/{src的basename}"；否则返回dst
        elif cls.isdir(src):
            if cls.exists(dst) and cover:
                cls.remove(dst)  # 原copytree遇到dst已存在时（无论是文件或文件夹），会引发FileExistsError
            return shutil.copytree(src, dst)  # 返回dst
        elif not cls.exists(src):
            raise FileNotFoundError(f"{src=} does not exist")
        else:
            raise TypeError(f"{src=} is neither file nor directory")

    @classmethod
    def json_dump(cls, obj: Any, path: str, ensure_ascii: bool = False, indent: Optional[int] = 4, *args,
                  **kwargs) -> None:
        with open(path, "w") as file:
            json.dump(obj, file, ensure_ascii=ensure_ascii, indent=indent, *args, **kwargs)

    @classmethod
    def json_load(cls, path: str, *args, **kwargs) -> Any:
        with open(path, "r") as file:
            return json.load(file, *args, **kwargs)


class String:
    @classmethod
    def backstep_format_params(cls, template, string):
        assert not re.search("}{", template)  # template的参数不能紧挨着

        keys = [m.group(1) for m in re.finditer(r"{(\w*)}", template)]

        values = list()
        for split in re.split(r"{\w*}", template):
            if not split:  # 关键字在开头或结尾时，会产生""
                continue

            value, string = string.split(split, 1)
            if value:
                values.append(value)

        if string:  # 关键字在结尾
            values.append(string)

        return dict(zip(keys, values))


@cache
class Database:
    def __init__(self, database: str):
        self.connection = sqlite3.connect(database)

    def query(self, sql: str, *args, **kwargs) -> pd.DataFrame:
        return pd.read_sql(sql, self.connection, *args, **kwargs)
