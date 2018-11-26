# -*- coding: utf-8 -*-

"""This module defines methods to create files from templates.
"""

import os

from jinja2 import Environment, PackageLoader, select_autoescape, BaseLoader, TemplateNotFound


class FileLoader(BaseLoader):
    """A loader that loads files"""

    def __init__(self):
        # type: () -> None
        BaseLoader.__init__(self)

    def get_source(self, environment, template):
        if not os.path.isfile(template):
            raise TemplateNotFound(template)

        mtime = os.path.getmtime(template)
        with open(template, 'r') as f:
            source = f.read()
        return source, template, lambda: mtime == os.path.getmtime(template)


def new_template_env(parent_package, tmp_folder):
    # type: (str, str) -> Environment
    return Environment(trim_blocks=True,
                       lstrip_blocks=True,
                       keep_trailing_newline=True,
                       autoescape=select_autoescape(default_for_string=False),
                       loader=PackageLoader(parent_package, package_path=tmp_folder),
                       enable_async=False,
                       )


def new_template_env_fs():
    # type: () -> Environment
    return Environment(trim_blocks=True,
                       lstrip_blocks=True,
                       keep_trailing_newline=True,
                       autoescape=select_autoescape(default_for_string=False),
                       loader=FileLoader(),
                       enable_async=False,
                       )
