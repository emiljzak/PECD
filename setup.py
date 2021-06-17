#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>

from setuptools import setup

copyright = 'Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>'
name = "PECD"
version = "0.9.dev0"
release = version
long_description = """
Original author:    Emil Zak <emil.zak@cfel.de>
Current maintainer: Emil Zak <emil.zak@cfel.de>
"""


setup(name=name,
      python_requires     = '>=3.6',
      author              = "Emil Zak, Andrey Yachmenev, Jochen Küpper",
      author_email        = "emil.zak@cfel.de",
      maintainer          = "Emil Zak",
      maintainer_email    = "emil.zak@cfel.de",
      url                 = "https://github.com/CFEL-CMI/PECD",
      description         = "Quantum-mechanical calculations of photo-electron circular dichroism",
      version             = version,
      long_description    = long_description,
      license             = "GPL",
      packages            = ['cmitemplate'],
      scripts             = ['scripts/cmitemplate_calc'],
      command_options={
          'build_sphinx': {
              'project': ('setup.py', name),
              'version': ('setup.py', version),
              'release': ('setup.py', release),
              'source_dir': ('setup.py', 'doc'),
              'copyright': ('setup.py', copyright)}
      },
      )
