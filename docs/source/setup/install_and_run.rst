Installing and Running BAG
==========================

This section describes how to install then run BAG along with Virtuoso.

Installation Requirements
-------------------------

BAG is compatible with Python 2.7+/3.5+, so you will need to have Python 2.7/3.5 installed.  For Linux/Unix systems, it
is recommended to install a separate Python distribution from the system Python.

BAG requires multiple Python packages, which are listed in the ``setup.py`` file in the source folder.  However, it is
strongly recommended to download `Anaconda Python <https://www.continuum.io/downloads>`_, which provides a Python
distribution with most of the packages preinstalled.  Otherwise, please refer to documentation for each required
package for how to install/build from source.

Installing Python
-----------------
In addition to the default packages that come with Anaconda (numpy, scipy, etc.), you'll need the following additional
packages:

- `python-future <https://pypi.python.org/pypi/future>`_

  This package provides Python 2/3 compatibility.  It is installable from ``pip``:

  .. code-block:: bash

      > pip install future

  ``pip`` also works for pre-downloaded tar file:

  .. code-block:: bash

      > pip install future-0.16.0.tar.gz

- `subprocess32 <https://pypi.python.org/pypi/subprocess32>`_ (Python 2 only)

  This package is a backport of Python 3.2's subprocess module to Python 2.  It is installable from ``pip``.

- `sqlitedict <https://pypi.python.org/pypi/sqlitedict>`_

  This is a dependency of OpenMDAO.  It is installable from ``pip``.

- `OpenMDAO <https://pypi.python.org/pypi/openmdao>`_

  This is a flexible optimization framework in Python developed by NASA.  It is installable from ``pip``.
  However, as of 2016 Jan. 23rd, the package on PyPI contains a unicode-related bug for Python 2.7.  a fixed
  version can be obtained from their git repo or from my Anaconda Cloud channel
  `here <https://anaconda.org/pkerichang/openmdao>`_.  You can install from my channel with the command:

  .. code-block:: bash

      > conda install --channel pkerichang openmdao

  similar to ``pip install``, ``conda install`` also works with pre-downloaded tar files.

- `mpich2 <https://anaconda.org/anaconda/mpich2>`_ (optional)

  This is the Message Passing Interface (MPI) library.  OpenMDAO and Pyoptsparse can optionally use this library
  for parallel computing.  You can install this package with:

  .. code-block:: bash

      > conda install mpich2

- `mpi4py <https://anaconda.org/anaconda/mpi4py>`_ (optional)

  This is the Python wrapper of ``mpich2``.  You can install this package with:

  .. code-block:: bash

      > conda install mpi4py

- `ipopt <https://anaconda.org/pkerichang/ipopt>`__ (optional)

  `Ipopt <https://projects.coin-or.org/Ipopt>`__ is a free software package for large-scale nonlinear optimization.
  This can be used to replace the default optimization solver that comes with scipy.  You can install this package with:

  .. code-block:: bash

      > conda install --channel pkerichang ipopt

- `pyoptsparse <https://anaconda.org/pkerichang/pyoptsparse>`_ (optional)

  ``pyoptsparse`` is a python package that contains a collection of optmization solvers, including a Python wrapper
  around ``Ipopt``.  You can install this package with:

  .. code-block:: bash

      > conda install --channel pkerichang pyoptsparse

Installing BAG
--------------

BAG is a standalone Python package.  To install BAG from source, simply go to the source directory, and type:

.. code-block:: bash

    > pip install .

BAG will then be installed to your Python distribution.  To uninstall BAG, simply type:

.. code-block:: bash

    > pip uninstall bag

.. _run_scripts:

Generating Virtuoso Scripts
---------------------------

BAG access and modifies schematics by communicating with a running Virtuoso program.  Therefore, Virtuoso needs to load
some skill function definitions and open a socket port for BAG to communicate with it.  BAG is packaged with the
scripts needed to perform these tasks.  To access these scripts:

#. Pick a directory to store these scripts.  Here we'll use the directory ``/foo/bar/baz/run_scripts``.
#. On the command line, run:

    .. code-block:: bash

        > python -m bag.virtuoso gen_scripts /foo/bar/baz/run_scripts

#. The directory ``/foo/bar/baz/run_scripts`` should now contain two files, ``start_bag.il`` and ``start_server.sh``.

Configuration Files
-------------------

BAG requires two YAML configuration files to start, ``bag_config.yaml`` and ``tech_config.yaml``.  Sample configuration
files should be available in the BAG source directory.  See sections :doc:`bag_config/bag_config` and
:doc:`tech_config/tech_config` to learn more about each setting in these files.

Environment Variable Setup
--------------------------

BAG needs three environment variables to be configured:

#. ``BAG_CONFIG_PATH``, the location of ``bag_config.yaml``.

#. ``BAG_WORK_DIR``, the working directory of the Virtuoso program.  BAG use this environment variable to find the
   socket port number used by Virtuoso.

#. ``BAG_TEMP_DIR``, the directory to save temporary data files to.  If this environment variable is not specified, BAG
   will save temporary files to ``/tmp``.

Virtuoso Startup
----------------

BAG communicates with a running Virtuoso program, so you need to start Virtuoso before start using BAG.  Once you start
Virtuoso, in the CIW window (the window that shows log messages), type:

.. code-block:: none

    load("/foo/bar/baz/run_scripts/start_bag.il")

This skill script will define some skill functions and start a BAG server process, which opens a socket port and
listen for BAG commands.  The server process ID is stored in the variable ``bag_proc``.  If you want to shut down this
server process, simply type the following in the CIW window:

.. code-block:: none

    ipcKillProcess(bag_proc)

However, once you kill this process, you need to load the skill script again before BAG can communicate with Virtuoso.

Running BAG
-----------

With the BAG server process (and optionally simulation server process) started, you can run BAG by simply starting
Python, then type:

.. code-block:: python

    >>> import bag
    >>> prj = bag.BagProject()

You can now use the :class:`~bag.BagProject` object to perform any tasks.
